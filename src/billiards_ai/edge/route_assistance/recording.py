"""Index authoritative shot effects into idempotent local record boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from billiards_ai.edge.route_assistance.records import PowerBand, ShotMediaRange
from billiards_ai.lab.field_assistance.live_state import ReplayTimeline


class ShotBoundaryKind(StrEnum):
    STARTED = "started"
    SETTLED = "settled"


@dataclass(frozen=True, slots=True)
class RecommendationPower:
    band: PowerBand
    approximate_percent: int

    def __post_init__(self) -> None:
        if (
            self.band
            not in {
                "light",
                "medium_light",
                "medium",
                "medium_firm",
                "firm",
            }
            or not 1 <= self.approximate_percent <= 100
        ):
            raise ValueError("recommendation power is invalid")


@dataclass(frozen=True, slots=True)
class ShotBoundary:
    kind: ShotBoundaryKind
    sequence: int
    timestamp_ms: int
    recommendation_id: str | None

    def __post_init__(self) -> None:
        if self.sequence <= 0 or self.timestamp_ms < 0:
            raise ValueError("shot boundary sequence and timestamp are invalid")
        if self.recommendation_id is not None and not self.recommendation_id.strip():
            raise ValueError("shot boundary recommendation ID is empty")


@dataclass(frozen=True, slots=True)
class ShotTimelineIndex:
    boundaries: tuple[ShotBoundary, ...]

    @classmethod
    def from_timeline(cls, timeline: ReplayTimeline) -> ShotTimelineIndex:
        boundaries: list[ShotBoundary] = []
        open_sequence: int | None = None
        open_timestamp_ms: int | None = None
        open_recommendation: str | None = None
        last_recommendation: str | None = None
        next_sequence = 1

        try:
            for frame in timeline.frames:
                recommendation = frame.payload.get("recommendation_id")
                if recommendation is not None:
                    if not isinstance(recommendation, str) or not recommendation.strip():
                        raise ValueError
                    last_recommendation = recommendation
                effects = frame.payload.get("effects", ())
                if not isinstance(effects, (list, tuple)) or not all(
                    isinstance(effect, str) for effect in effects
                ):
                    raise ValueError
                if effects.count("SHOT_STARTED") > 1 or effects.count("SHOT_SETTLED") > 1:
                    raise ValueError
                for effect in effects:
                    if effect == "SHOT_STARTED":
                        sequence = int(frame.payload["shot_sequence"]) + 1
                        if open_sequence is not None or sequence != next_sequence:
                            raise ValueError
                        open_sequence = sequence
                        open_timestamp_ms = frame.timestamp_ms
                        open_recommendation = last_recommendation
                    elif effect == "SHOT_SETTLED":
                        sequence = int(frame.payload["shot_sequence"])
                        if open_sequence != sequence or open_timestamp_ms is None:
                            raise ValueError
                        boundaries.extend(
                            (
                                ShotBoundary(
                                    kind=ShotBoundaryKind.STARTED,
                                    sequence=sequence,
                                    timestamp_ms=open_timestamp_ms,
                                    recommendation_id=open_recommendation,
                                ),
                                ShotBoundary(
                                    kind=ShotBoundaryKind.SETTLED,
                                    sequence=sequence,
                                    timestamp_ms=frame.timestamp_ms,
                                    recommendation_id=open_recommendation,
                                ),
                            )
                        )
                        open_sequence = None
                        open_timestamp_ms = None
                        open_recommendation = None
                        next_sequence += 1
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("authoritative shot effects are invalid") from exc
        return cls(boundaries=tuple(boundaries))

    def boundaries_between(
        self,
        *,
        after_ms: int,
        through_ms: int,
    ) -> tuple[ShotBoundary, ...]:
        if after_ms < -1 or through_ms < 0 or after_ms > through_ms:
            raise ValueError("shot boundary window is invalid")
        return tuple(
            boundary
            for boundary in self.boundaries
            if after_ms < boundary.timestamp_ms <= through_ms
        )


def build_media_range(
    *,
    artifact_id: str,
    started_ms: int,
    settled_ms: int,
    duration_ms: int,
    padding_ms: int = 2_000,
) -> ShotMediaRange:
    """Build a development presentation range around one authoritative shot."""

    if (
        duration_ms <= 0
        or padding_ms < 0
        or started_ms < 0
        or settled_ms <= started_ms
        or settled_ms > duration_ms
    ):
        raise ValueError("authoritative shot timing is invalid")
    return ShotMediaRange(
        artifact_id=artifact_id,
        source_locator="presentation.mp4",
        source_kind="development_replay",
        start_ms=max(0, started_ms - padding_ms),
        end_ms=min(duration_ms, settled_ms + padding_ms),
    )


def load_recommendation_power(path: Path) -> dict[str, RecommendationPower]:
    """Load generic power facts from one verified replay recommendation file."""

    source = path.resolve()
    if path.is_symlink() or not source.is_file():
        raise ValueError("recommendation power source is missing or unsafe")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError
        generations = payload["generations"]
        if not isinstance(generations, list):
            raise ValueError
        result: dict[str, RecommendationPower] = {}
        for generation in generations:
            display = generation.get("display")
            if display is None:
                continue
            recommendation_id = display["recommendation_id"]
            power = display["power"]
            if (
                not isinstance(recommendation_id, str)
                or not recommendation_id.strip()
                or recommendation_id in result
                or power.get("personalized") is not False
            ):
                raise ValueError
            result[recommendation_id] = RecommendationPower(
                band=power["band"],
                approximate_percent=int(power["approximate_percent"]),
            )
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("recommendation power evidence is invalid") from exc
