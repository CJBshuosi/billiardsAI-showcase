"""Experimental shot-local Kalman tracks and delayed collision decisions.

This module is deliberately independent from the legacy motion-fragment
observer.  It keeps one short-lived track per pre-shot anchor, associates a
whole frame at once, and waits for post-contact motion before admitting a
collision.  It is research evidence, not a global physical-ball identity
contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
    ShotAnchor,
    ShotLocalEvent,
)


@dataclass(frozen=True, slots=True)
class ActiveTrackConfig:
    position_gate_mm: float
    unmatched_cost: float
    process_noise_mm: float
    measurement_noise_mm: float
    max_prediction_frames: int
    appearance_weight: float
    group_mismatch_cost: float
    collision_distance_mm: float
    moving_speed_mm_per_frame: float
    stationary_speed_mm_per_frame: float
    collision_lookahead_frames: int
    collision_candidate_margin: float

    def __post_init__(self) -> None:
        positive = (
            self.position_gate_mm,
            self.unmatched_cost,
            self.process_noise_mm,
            self.measurement_noise_mm,
            self.collision_distance_mm,
            self.moving_speed_mm_per_frame,
            self.stationary_speed_mm_per_frame,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("active-track distances and costs must be positive")
        if self.max_prediction_frames < 0 or self.collision_lookahead_frames < 1:
            raise ValueError("active-track frame limits are invalid")
        if self.appearance_weight < 0 or self.group_mismatch_cost < 0:
            raise ValueError("active-track penalties must be non-negative")
        if not 0 <= self.collision_candidate_margin <= 1:
            raise ValueError("collision candidate margin must be between zero and one")


@dataclass(frozen=True, slots=True)
class ActiveTrackSample:
    frame_id: int
    timestamp_ms: int
    x_mm: float
    y_mm: float
    vx_mm_per_frame: float
    vy_mm_per_frame: float
    state: str
    confidence: float
    source_detection_id: str | None

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("active-track sample time must be non-negative")
        if self.state not in {"observed", "predicted", "reacquired"}:
            raise ValueError("active-track sample state is invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("active-track confidence must be between zero and one")
        if self.state == "predicted" and self.source_detection_id is not None:
            raise ValueError("predicted sample cannot claim a source detection")
        if self.state != "predicted" and not self.source_detection_id:
            raise ValueError("observed sample requires a source detection")


@dataclass(frozen=True, slots=True)
class ActiveBallTrack:
    shot_id: str
    track_id: str
    group: BallGroup
    samples: tuple[ActiveTrackSample, ...]

    def __post_init__(self) -> None:
        if not self.track_id.startswith(f"{self.shot_id}-anchor-"):
            raise ValueError("active track must retain one pre-shot anchor ID")
        frame_ids = [sample.frame_id for sample in self.samples]
        if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("active-track samples must use unique ordered frames")


@dataclass(frozen=True, slots=True)
class DelayedCollisionDecision:
    actor_id: str
    target_id: str | None
    contact_frame_id: int
    decision_frame_id: int
    status: str
    confidence: float
    evidence_frame_ids: tuple[int, ...]
    reason: str | None = None
    detection_layer: str = "legacy"
    contact_kind: str = "legacy_transfer"
    minimum_distance_mm: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"observed", "abstained"}:
            raise ValueError("delayed collision status is invalid")
        if self.decision_frame_id < self.contact_frame_id:
            raise ValueError("collision decision cannot precede contact")
        if not self.evidence_frame_ids:
            raise ValueError("collision decision requires evidence frames")
        if not 0 <= self.confidence <= 1:
            raise ValueError("collision confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class CushionContactDecision:
    actor_id: str
    rail: str
    contact_frame_id: int
    decision_frame_id: int
    detection_layer: str
    confidence: float
    minimum_distance_mm: float
    contact_kind: str = "rail_contact"
    evidence_frame_ids: tuple[int, ...] = ()


def observe_three_layer_cushions(
    tracks: Sequence[ActiveBallTrack],
    *,
    table_length_mm: float,
    table_width_mm: float,
    contact_distance_mm: float,
    normal_motion_min_mm: float = 3.0,
    max_observation_gap_frames: int = 3,
    sparse_extra_mm: float = 40.0,
    ball_collisions: Sequence[DelayedCollisionDecision] = (),
    trajectory_max_gap_frames: int = 10,
    trajectory_boundary_tolerance_mm: float = 90.0,
    trajectory_min_turn_degrees: float = 25.0,
    trajectory_max_speed_mm_per_frame: float = 120.0,
) -> tuple[CushionContactDecision, ...]:
    """Observe direct, frame-skipped, and sustained cushion contacts."""

    if (
        table_length_mm <= 0
        or table_width_mm <= 0
        or contact_distance_mm <= 0
        or normal_motion_min_mm <= 0
        or max_observation_gap_frames < 1
        or sparse_extra_mm < 0
        or trajectory_max_gap_frames < max_observation_gap_frames
        or trajectory_boundary_tolerance_mm <= 0
        or not 0 < trajectory_min_turn_degrees < 180
        or trajectory_max_speed_mm_per_frame <= 0
    ):
        raise ValueError("cushion geometry must be positive")

    def rail_distance(sample: ActiveTrackSample, rail: str) -> float:
        if rail == "left":
            return sample.x_mm
        if rail == "right":
            return table_length_mm - sample.x_mm
        if rail == "bottom":
            return sample.y_mm
        return table_width_mm - sample.y_mm

    decisions: list[CushionContactDecision] = []
    for track in tracks:
        emitted: set[str] = set()
        ordered = sorted(track.samples, key=lambda item: item.frame_id)
        observed = [sample for sample in ordered if sample.state != "predicted"]
        for first, middle, last in zip(observed, observed[1:], observed[2:], strict=False):
            first_gap = middle.frame_id - first.frame_id
            last_gap = last.frame_id - middle.frame_id
            if (
                first_gap <= 0
                or last_gap <= 0
                or first_gap > max_observation_gap_frames
                or last_gap > max_observation_gap_frames
            ):
                continue
            sparse = first_gap > 1 or last_gap > 1
            proximity = contact_distance_mm + (sparse_extra_mm if sparse else 0.0)
            candidates: list[tuple[float, str]] = []
            for rail in ("left", "right", "bottom", "top"):
                before = rail_distance(first, rail)
                nearest = rail_distance(middle, rail)
                after = rail_distance(last, rail)
                if (
                    rail not in emitted
                    and nearest <= proximity
                    and nearest - before <= -normal_motion_min_mm
                    and after - nearest >= normal_motion_min_mm
                ):
                    candidates.append((nearest, rail))
            candidates.sort()
            if not candidates or (
                len(candidates) > 1 and math.isclose(candidates[0][0], candidates[1][0])
            ):
                continue
            nearest, rail = candidates[0]
            layer = "L2" if sparse or nearest > contact_distance_mm else "L1"
            decisions.append(
                CushionContactDecision(
                    actor_id=track.track_id,
                    rail=rail,
                    contact_frame_id=middle.frame_id,
                    decision_frame_id=last.frame_id,
                    detection_layer=layer,
                    confidence={"L1": 0.98, "L2": 0.84}[layer],
                    minimum_distance_mm=float(nearest),
                )
            )
            emitted.add(rail)
        for first, middle, last in zip(observed, observed[1:], observed[2:], strict=False):
            first_gap = middle.frame_id - first.frame_id
            last_gap = last.frame_id - middle.frame_id
            if (
                first_gap <= 0
                or last_gap <= 0
                or first_gap > trajectory_max_gap_frames
                or last_gap > trajectory_max_gap_frames
            ):
                continue
            inbound = np.asarray((middle.x_mm - first.x_mm, middle.y_mm - first.y_mm))
            outbound = np.asarray((last.x_mm - middle.x_mm, last.y_mm - middle.y_mm))
            inbound_distance = float(np.linalg.norm(inbound))
            outbound_distance = float(np.linalg.norm(outbound))
            if (
                inbound_distance < normal_motion_min_mm
                or outbound_distance < normal_motion_min_mm
                or inbound_distance / first_gap > trajectory_max_speed_mm_per_frame
                or outbound_distance / last_gap > trajectory_max_speed_mm_per_frame
            ):
                continue
            cosine = float(inbound @ outbound) / (inbound_distance * outbound_distance)
            turn_degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if turn_degrees < trajectory_min_turn_degrees:
                continue
            if any(
                collision.status == "observed"
                and track.track_id in {collision.actor_id, collision.target_id}
                and first.frame_id <= collision.contact_frame_id <= last.frame_id
                for collision in ball_collisions
            ):
                continue

            x_rail = "left" if middle.x_mm <= table_length_mm - middle.x_mm else "right"
            y_rail = "bottom" if middle.y_mm <= table_width_mm - middle.y_mm else "top"
            x_distance = rail_distance(middle, x_rail)
            y_distance = rail_distance(middle, y_rail)
            corner = f"{x_rail}-{y_rail}"
            if (
                x_distance <= trajectory_boundary_tolerance_mm
                and y_distance <= trajectory_boundary_tolerance_mm
            ):
                rail = f"pocket-jaw:{corner}"
                emitted_rails = {x_rail, y_rail}
                minimum_distance = min(x_distance, y_distance)
                boundary_before = math.hypot(
                    rail_distance(first, x_rail), rail_distance(first, y_rail)
                )
                boundary_middle = math.hypot(x_distance, y_distance)
                boundary_after = math.hypot(
                    rail_distance(last, x_rail), rail_distance(last, y_rail)
                )
            else:
                minimum_distance, rail = min(
                    (rail_distance(middle, candidate), candidate)
                    for candidate in ("left", "right", "bottom", "top")
                )
                if minimum_distance > trajectory_boundary_tolerance_mm:
                    continue
                emitted_rails = {rail}
                boundary_before = rail_distance(first, rail)
                boundary_middle = rail_distance(middle, rail)
                boundary_after = rail_distance(last, rail)
            if emitted.intersection(emitted_rails):
                continue
            if (
                boundary_before - boundary_middle < normal_motion_min_mm
                or boundary_after - boundary_middle < normal_motion_min_mm
            ):
                continue
            decisions.append(
                CushionContactDecision(
                    actor_id=track.track_id,
                    rail=rail,
                    contact_frame_id=middle.frame_id,
                    decision_frame_id=last.frame_id,
                    detection_layer="L3",
                    confidence=0.78,
                    minimum_distance_mm=float(minimum_distance),
                    contact_kind="trajectory_inferred_rebound",
                    evidence_frame_ids=(first.frame_id, middle.frame_id, last.frame_id),
                )
            )
            emitted.update(emitted_rails)
        for split in range(2, len(observed) - 3):
            inbound_samples = observed[split - 2 : split + 1]
            outbound_samples = observed[split + 1 : split + 4]
            evidence = (*inbound_samples, *outbound_samples)
            if any(
                current.frame_id - previous.frame_id > trajectory_max_gap_frames
                for previous, current in zip(evidence, evidence[1:], strict=False)
            ):
                continue

            def fitted_velocity(samples: Sequence[ActiveTrackSample]) -> np.ndarray:
                frames = np.asarray([sample.frame_id for sample in samples], dtype=np.float64)
                centered = frames - frames.mean()
                denominator = float(centered @ centered)
                return np.asarray(
                    (
                        float(centered @ np.asarray([sample.x_mm for sample in samples]))
                        / denominator,
                        float(centered @ np.asarray([sample.y_mm for sample in samples]))
                        / denominator,
                    )
                )

            inbound = fitted_velocity(inbound_samples)
            outbound = fitted_velocity(outbound_samples)
            inbound_speed = float(np.linalg.norm(inbound))
            outbound_speed = float(np.linalg.norm(outbound))
            if (
                not normal_motion_min_mm <= inbound_speed <= trajectory_max_speed_mm_per_frame
                or not normal_motion_min_mm <= outbound_speed <= trajectory_max_speed_mm_per_frame
            ):
                continue
            cosine = float(inbound @ outbound) / (inbound_speed * outbound_speed)
            turn_degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            if turn_degrees < trajectory_min_turn_degrees:
                continue
            first, last = evidence[0], evidence[-1]
            if any(
                collision.status == "observed"
                and track.track_id in {collision.actor_id, collision.target_id}
                and first.frame_id <= collision.contact_frame_id <= last.frame_id
                for collision in ball_collisions
            ):
                continue
            contact = min(
                (observed[split], observed[split + 1]),
                key=lambda sample: min(
                    rail_distance(sample, candidate)
                    for candidate in ("left", "right", "bottom", "top")
                ),
            )
            x_rail = "left" if contact.x_mm <= table_length_mm - contact.x_mm else "right"
            y_rail = "bottom" if contact.y_mm <= table_width_mm - contact.y_mm else "top"
            x_distance = rail_distance(contact, x_rail)
            y_distance = rail_distance(contact, y_rail)
            if (
                x_distance <= trajectory_boundary_tolerance_mm
                and y_distance <= trajectory_boundary_tolerance_mm
            ):
                rail = f"pocket-jaw:{x_rail}-{y_rail}"
                emitted_rails = {x_rail, y_rail}
                minimum_distance = min(x_distance, y_distance)
                boundary_before = math.hypot(
                    rail_distance(first, x_rail), rail_distance(first, y_rail)
                )
                boundary_contact = math.hypot(x_distance, y_distance)
                boundary_after = math.hypot(
                    rail_distance(last, x_rail), rail_distance(last, y_rail)
                )
            else:
                minimum_distance, rail = min(
                    (rail_distance(contact, candidate), candidate)
                    for candidate in ("left", "right", "bottom", "top")
                )
                if minimum_distance > trajectory_boundary_tolerance_mm:
                    continue
                emitted_rails = {rail}
                boundary_before = rail_distance(first, rail)
                boundary_contact = rail_distance(contact, rail)
                boundary_after = rail_distance(last, rail)
            if emitted.intersection(emitted_rails):
                continue
            if (
                boundary_before - boundary_contact < normal_motion_min_mm
                or boundary_after - boundary_contact < normal_motion_min_mm
            ):
                continue
            decisions.append(
                CushionContactDecision(
                    actor_id=track.track_id,
                    rail=rail,
                    contact_frame_id=contact.frame_id,
                    decision_frame_id=last.frame_id,
                    detection_layer="L3",
                    confidence=0.82,
                    minimum_distance_mm=float(minimum_distance),
                    contact_kind="trajectory_fitted_rebound",
                    evidence_frame_ids=tuple(sample.frame_id for sample in evidence),
                )
            )
            emitted.update(emitted_rails)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            rails = {
                "left": (
                    previous.x_mm,
                    current.x_mm,
                    previous.vx_mm_per_frame,
                    current.vx_mm_per_frame,
                ),
                "right": (
                    table_length_mm - previous.x_mm,
                    table_length_mm - current.x_mm,
                    -previous.vx_mm_per_frame,
                    -current.vx_mm_per_frame,
                ),
                "bottom": (
                    previous.y_mm,
                    current.y_mm,
                    previous.vy_mm_per_frame,
                    current.vy_mm_per_frame,
                ),
                "top": (
                    table_width_mm - previous.y_mm,
                    table_width_mm - current.y_mm,
                    -previous.vy_mm_per_frame,
                    -current.vy_mm_per_frame,
                ),
            }
            for rail, (
                before_distance,
                current_distance,
                before_normal,
                current_normal,
            ) in rails.items():
                if rail in emitted:
                    continue
                layer: str | None = None
                minimum_distance = min(before_distance, current_distance)
                if before_distance > contact_distance_mm >= current_distance:
                    layer = "L1"
                elif (
                    before_distance > contact_distance_mm
                    and current_distance > contact_distance_mm
                    and before_normal < 0 < current_normal
                    and before_distance + before_normal <= contact_distance_mm
                ):
                    layer = "L2"
                    minimum_distance = max(0.0, before_distance + before_normal)
                elif (
                    before_distance <= contact_distance_mm
                    and current_distance > contact_distance_mm
                    and current_normal > 1.0
                ) or (
                    before_distance <= contact_distance_mm
                    and current_distance <= contact_distance_mm
                    and before_normal < 0 < current_normal
                ):
                    layer = "L3"
                if layer is None:
                    continue
                decisions.append(
                    CushionContactDecision(
                        actor_id=track.track_id,
                        rail=rail,
                        contact_frame_id=current.frame_id,
                        decision_frame_id=current.frame_id,
                        detection_layer=layer,
                        confidence={"L1": 0.98, "L2": 0.84, "L3": 0.74}[layer],
                        minimum_distance_mm=float(minimum_distance),
                    )
                )
                emitted.add(rail)
    return tuple(
        sorted(decisions, key=lambda item: (item.contact_frame_id, item.actor_id, item.rail))
    )


def minimum_cost_assignment(costs: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Return the globally minimum-cost column for every matrix row.

    This is the rectangular Hungarian algorithm.  Callers add private dummy
    columns when an unmatched row is allowed.
    """

    if not costs:
        return ()
    row_count = len(costs)
    column_count = len(costs[0])
    if column_count < row_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment matrix must be rectangular with enough columns")
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = math.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    float(costs[current_row - 1][column - 1])
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            if math.isinf(delta):
                raise ValueError("assignment matrix has no complete solution")
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while current_column:
            previous = predecessor[current_column]
            matched_row[current_column] = matched_row[previous]
            current_column = previous
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return tuple(assignment)


def assign_calibrated_points(
    predicted: Sequence[tuple[float, float]],
    observed: Sequence[tuple[float, float]],
    *,
    maximum_distance_mm: float,
) -> tuple[tuple[int, int], ...]:
    """Globally associate point observations while allowing either side to remain unmatched."""

    if not math.isfinite(maximum_distance_mm) or maximum_distance_mm <= 0:
        raise ValueError("maximum assignment distance must be finite and positive")
    if not predicted or not observed:
        return ()
    rejected_cost = maximum_distance_mm * 4.0
    costs: list[list[float]] = []
    for predicted_index, predicted_point in enumerate(predicted):
        row = [
            (
                distance
                if (distance := math.dist(predicted_point, observed_point)) <= maximum_distance_mm
                else rejected_cost
            )
            for observed_point in observed
        ]
        row.extend(
            maximum_distance_mm if dummy == predicted_index else rejected_cost
            for dummy in range(len(predicted))
        )
        costs.append(row)
    assignments = minimum_cost_assignment(costs)
    return tuple(
        (predicted_index, observed_index)
        for predicted_index, observed_index in enumerate(assignments)
        if observed_index < len(observed)
        and costs[predicted_index][observed_index] <= maximum_distance_mm
    )


def _appearance_distance(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.0
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(first @ second) / denominator))


def extract_ball_appearance_signatures(
    images_by_frame_id: Mapping[int, np.ndarray],
    frames: Sequence[ObservationFrame],
) -> dict[str, np.ndarray]:
    """Extract compact colour/texture evidence from detector boxes.

    The inner ellipse deliberately excludes most green cloth and box edges.
    These signatures are only an association cue; they never assign a numbered
    ball identity or override an impossible motion gate.
    """

    signatures: dict[str, np.ndarray] = {}
    for frame in frames:
        image = images_by_frame_id.get(frame.frame_id)
        if image is None or image.ndim != 3:
            continue
        image_height, image_width = image.shape[:2]
        for observation in frame.observations:
            if observation.bbox_xyxy_px is None:
                continue
            x1, y1, x2, y2 = observation.bbox_xyxy_px
            x1, x2 = max(0, x1), min(image_width, x2)
            y1, y2 = max(0, y1), min(image_height, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = image[y1:y2, x1:x2]
            yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
            center_x, center_y = (crop.shape[1] - 1) / 2, (crop.shape[0] - 1) / 2
            radius_x, radius_y = max(1.0, crop.shape[1] * 0.4), max(1.0, crop.shape[0] * 0.4)
            mask = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1
            pixels = crop[mask]
            if not pixels.size:
                continue
            histograms = [
                np.histogram(pixels[:, channel], bins=8, range=(0, 256))[0] for channel in range(3)
            ]
            channel_means = pixels.mean(axis=0) / 255.0
            signature = np.concatenate((*histograms, channel_means * len(pixels))).astype(
                np.float64
            )
            norm = float(np.linalg.norm(signature))
            if norm > 1e-12:
                signatures[observation.detection_id] = signature / norm
    return signatures


@dataclass(slots=True)
class _KalmanTrack:
    anchor: ShotAnchor
    state: np.ndarray
    covariance: np.ndarray
    samples: list[ActiveTrackSample] = field(default_factory=list)
    missed_frames: int = 0
    appearance: np.ndarray | None = None

    def predict(self, config: ActiveTrackConfig) -> None:
        transition = np.asarray(
            ((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        )
        process_variance = config.process_noise_mm**2
        process = np.diag(
            (process_variance, process_variance, process_variance / 4, process_variance / 4)
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

    def innovation_distance(
        self,
        observation: CalibratedBallObservation,
        config: ActiveTrackConfig,
    ) -> float:
        delta = np.asarray((observation.x_mm, observation.y_mm), dtype=np.float64) - self.state[:2]
        measurement_covariance = self.covariance[:2, :2] + np.eye(2) * (
            config.measurement_noise_mm**2
        )
        try:
            mahalanobis = float(
                math.sqrt(max(0.0, delta @ np.linalg.solve(measurement_covariance, delta)))
            )
        except np.linalg.LinAlgError:
            return math.inf
        return mahalanobis

    def update(
        self,
        observation: CalibratedBallObservation,
        config: ActiveTrackConfig,
        appearance: np.ndarray | None,
    ) -> None:
        observation_matrix = np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)), dtype=np.float64
        )
        measurement = np.asarray((observation.x_mm, observation.y_mm), dtype=np.float64)
        noise = np.eye(2) * (config.measurement_noise_mm**2)
        innovation = measurement - observation_matrix @ self.state
        innovation_covariance = observation_matrix @ self.covariance @ observation_matrix.T + noise
        gain = self.covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain @ observation_matrix) @ self.covariance
        if appearance is not None:
            self.appearance = (
                appearance if self.appearance is None else self.appearance * 0.8 + appearance * 0.2
            )


def _observation_cost(
    track: _KalmanTrack,
    observation: CalibratedBallObservation,
    config: ActiveTrackConfig,
    appearance: np.ndarray | None,
) -> float:
    euclidean = math.hypot(
        observation.x_mm - float(track.state[0]), observation.y_mm - float(track.state[1])
    )
    if euclidean > config.position_gate_mm:
        return config.unmatched_cost * 4
    group_penalty = 0.0
    if (
        track.anchor.group is not BallGroup.UNKNOWN
        and observation.group is not BallGroup.UNKNOWN
        and track.anchor.group is not observation.group
    ):
        group_penalty = config.group_mismatch_cost
    appearance_penalty = config.appearance_weight * _appearance_distance(
        track.appearance, appearance
    )
    uncertainty_cost = track.innovation_distance(observation, config)
    return euclidean + uncertainty_cost + group_penalty + appearance_penalty


def track_anchor_roster(
    frames: Sequence[ObservationFrame],
    anchors: Sequence[ShotAnchor],
    config: ActiveTrackConfig,
    *,
    appearance_by_detection_id: Mapping[str, Sequence[float]] | None = None,
) -> tuple[ActiveBallTrack, ...]:
    """Track one shot-local anchor roster with Kalman prediction and global assignment."""

    if not anchors:
        return ()
    shot_ids = {anchor.shot_id for anchor in anchors}
    if len(shot_ids) != 1:
        raise ValueError("active-track anchors must belong to one shot")
    appearance_rows = appearance_by_detection_id or {}
    tracks = [
        _KalmanTrack(
            anchor=anchor,
            state=np.asarray((anchor.x_mm, anchor.y_mm, 0.0, 0.0), dtype=np.float64),
            covariance=np.diag((25.0, 25.0, 400.0, 400.0)),
            appearance=(
                None
                if anchor.source_detection_id not in appearance_rows
                else np.asarray(appearance_rows[anchor.source_detection_id], dtype=np.float64)
            ),
        )
        for anchor in sorted(anchors, key=lambda item: item.anchor_id)
    ]
    for frame in sorted(frames, key=lambda item: item.frame_id):
        for track in tracks:
            track.predict(config)
        observations = tuple(sorted(frame.observations, key=lambda item: item.detection_id))
        costs: list[list[float]] = []
        for track_index, track in enumerate(tracks):
            row = [
                _observation_cost(
                    track,
                    observation,
                    config,
                    (
                        None
                        if observation.detection_id not in appearance_rows
                        else np.asarray(appearance_rows[observation.detection_id], dtype=np.float64)
                    ),
                )
                for observation in observations
            ]
            row.extend(
                config.unmatched_cost if dummy == track_index else config.unmatched_cost * 3
                for dummy in range(len(tracks))
            )
            costs.append(row)
        assignments = minimum_cost_assignment(costs)
        for track_index, track in enumerate(tracks):
            column = assignments[track_index]
            matched = (
                column < len(observations) and costs[track_index][column] < config.unmatched_cost
            )
            if matched:
                observation = observations[column]
                previous_misses = track.missed_frames
                appearance = (
                    None
                    if observation.detection_id not in appearance_rows
                    else np.asarray(appearance_rows[observation.detection_id], dtype=np.float64)
                )
                track.update(observation, config, appearance)
                track.missed_frames = 0
                track.samples.append(
                    ActiveTrackSample(
                        frame_id=frame.frame_id,
                        timestamp_ms=frame.timestamp_ms,
                        x_mm=float(track.state[0]),
                        y_mm=float(track.state[1]),
                        vx_mm_per_frame=float(track.state[2]),
                        vy_mm_per_frame=float(track.state[3]),
                        state="reacquired" if previous_misses else "observed",
                        confidence=observation.confidence,
                        source_detection_id=observation.detection_id,
                    )
                )
                continue
            track.missed_frames += 1
            if track.missed_frames <= config.max_prediction_frames:
                track.samples.append(
                    ActiveTrackSample(
                        frame_id=frame.frame_id,
                        timestamp_ms=frame.timestamp_ms,
                        x_mm=float(track.state[0]),
                        y_mm=float(track.state[1]),
                        vx_mm_per_frame=float(track.state[2]),
                        vy_mm_per_frame=float(track.state[3]),
                        state="predicted",
                        confidence=max(0.0, track.anchor.confidence * 0.7**track.missed_frames),
                        source_detection_id=None,
                    )
                )
    shot_id = next(iter(shot_ids))
    return tuple(
        ActiveBallTrack(
            shot_id=shot_id,
            track_id=track.anchor.anchor_id,
            group=track.anchor.group,
            samples=tuple(track.samples),
        )
        for track in tracks
    )


def _speed(first: ActiveTrackSample, second: ActiveTrackSample) -> float:
    step = second.frame_id - first.frame_id
    if step <= 0:
        return math.inf
    return math.hypot(second.x_mm - first.x_mm, second.y_mm - first.y_mm) / step


def _sample_map(track: ActiveBallTrack) -> dict[int, ActiveTrackSample]:
    return {sample.frame_id: sample for sample in track.samples}


def _distance(first: ActiveTrackSample, second: ActiveTrackSample) -> float:
    return math.hypot(first.x_mm - second.x_mm, first.y_mm - second.y_mm)


def _swept_minimum_distance(
    actor_before: ActiveTrackSample,
    actor_after: ActiveTrackSample,
    target_before: ActiveTrackSample,
    target_after: ActiveTrackSample,
) -> float:
    relative_start = np.asarray(
        (actor_before.x_mm - target_before.x_mm, actor_before.y_mm - target_before.y_mm),
        dtype=np.float64,
    )
    relative_end = np.asarray(
        (actor_after.x_mm - target_after.x_mm, actor_after.y_mm - target_after.y_mm),
        dtype=np.float64,
    )
    relative_step = relative_end - relative_start
    denominator = float(relative_step @ relative_step)
    if denominator <= 1e-12:
        return float(np.linalg.norm(relative_start))
    fraction = max(0.0, min(1.0, -float(relative_start @ relative_step) / denominator))
    return float(np.linalg.norm(relative_start + fraction * relative_step))


def _pair_candidate(
    actor: ActiveBallTrack,
    target: ActiveBallTrack,
    *,
    after_frame_id: int,
    config: ActiveTrackConfig,
) -> DelayedCollisionDecision | None:
    actor_samples = _sample_map(actor)
    target_samples = _sample_map(target)
    shared = sorted(set(actor_samples) & set(target_samples))
    noise_motion = max(1.0, config.measurement_noise_mm)
    for previous_frame, current_frame in zip(shared, shared[1:], strict=False):
        if current_frame <= after_frame_id or current_frame != previous_frame + 1:
            continue
        actor_before, actor_current = actor_samples[previous_frame], actor_samples[current_frame]
        target_before = target_samples[previous_frame]
        target_current = target_samples[current_frame]
        previous_distance = _distance(actor_before, target_before)
        current_distance = _distance(actor_current, target_current)
        both_current_observed = (
            actor_current.state != "predicted" and target_current.state != "predicted"
        )
        if (
            previous_distance > config.collision_distance_mm
            and current_distance <= config.collision_distance_mm
            and both_current_observed
        ):
            return DelayedCollisionDecision(
                actor_id=actor.track_id,
                target_id=target.track_id,
                contact_frame_id=current_frame,
                decision_frame_id=current_frame,
                status="observed",
                confidence=0.98,
                evidence_frame_ids=(previous_frame, current_frame),
                detection_layer="L1",
                contact_kind="new_contact",
                minimum_distance_mm=current_distance,
            )
        swept_distance = _swept_minimum_distance(
            actor_before, actor_current, target_before, target_current
        )
        if (
            previous_distance > config.collision_distance_mm
            and current_distance > config.collision_distance_mm
            and swept_distance <= config.collision_distance_mm
            and both_current_observed
        ):
            return DelayedCollisionDecision(
                actor_id=actor.track_id,
                target_id=target.track_id,
                contact_frame_id=current_frame,
                decision_frame_id=current_frame,
                status="observed",
                confidence=0.84,
                evidence_frame_ids=(previous_frame, current_frame),
                detection_layer="L2",
                contact_kind="swept_contact",
                minimum_distance_mm=swept_distance,
            )
        prior_frame = previous_frame - 1
        if (
            previous_distance <= config.collision_distance_mm
            and current_distance <= config.collision_distance_mm
            and target_current.state != "predicted"
            and prior_frame in target_samples
        ):
            prior_target_speed = _speed(target_samples[prior_frame], target_before)
            current_target_speed = _speed(target_before, target_current)
            if (
                current_target_speed > noise_motion
                and current_target_speed > prior_target_speed + noise_motion / 2
            ):
                decision_frame = min(
                    current_frame + config.collision_lookahead_frames,
                    max(shared),
                )
                if decision_frame <= current_frame:
                    continue
                return DelayedCollisionDecision(
                    actor_id=actor.track_id,
                    target_id=target.track_id,
                    contact_frame_id=current_frame,
                    decision_frame_id=decision_frame,
                    status="observed",
                    confidence=0.74,
                    evidence_frame_ids=tuple(range(prior_frame, decision_frame + 1)),
                    detection_layer="L3",
                    contact_kind="sustained_contact_impulse",
                    minimum_distance_mm=min(previous_distance, current_distance),
                )
    return None


def observe_delayed_collision_chain(
    tracks: Sequence[ActiveBallTrack],
    first_contact: ShotLocalEvent,
    config: ActiveTrackConfig,
) -> tuple[DelayedCollisionDecision, ...]:
    """Build a causal contact graph with direct, swept, then delayed evidence."""

    if (
        first_contact.event_type != "first_contact"
        or first_contact.status != "observed"
        or first_contact.target_id is None
    ):
        return ()
    by_id = {track.track_id: track for track in tracks}
    if first_contact.target_id not in by_id:
        return ()
    activated_at = {first_contact.target_id: max(first_contact.evidence_frame_ids)}
    emitted_pairs: set[frozenset[str]] = set()
    decisions: list[DelayedCollisionDecision] = []
    while True:
        candidates: list[DelayedCollisionDecision] = []
        for actor_id, active_frame in sorted(activated_at.items()):
            for target_id, target_track in sorted(by_id.items()):
                pair = frozenset((actor_id, target_id))
                if target_id == actor_id or pair in emitted_pairs or target_id in activated_at:
                    continue
                candidate = _pair_candidate(
                    by_id[actor_id], target_track, after_frame_id=active_frame, config=config
                )
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda item: (
                item.contact_frame_id,
                {"L1": 0, "L2": 1, "L3": 2}[item.detection_layer],
                -item.confidence,
                item.actor_id,
                item.target_id or "",
            ),
        )
        decisions.append(best)
        if best.target_id is None:
            break
        emitted_pairs.add(frozenset((best.actor_id, best.target_id)))
        activated_at[best.target_id] = best.contact_frame_id
    return tuple(sorted(decisions, key=lambda item: item.contact_frame_id))
