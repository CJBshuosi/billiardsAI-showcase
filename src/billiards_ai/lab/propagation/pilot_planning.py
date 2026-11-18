"""Immutable adaptive window planning and exact-start relay evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, computed_field, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.contracts import (
    InstanceAbsence,
    PropagationResult,
    PropagationWindow,
)
from billiards_ai.lab.propagation.windows import plan_windows


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


class WindowAttempt(FrozenContract):
    window: PropagationWindow
    status: Literal["planned", "committed", "failed", "superseded"] = "planned"
    failure_code: NonEmptyStr | None = None

    @model_validator(mode="after")
    def failure_state_is_explicit(self) -> WindowAttempt:
        if self.status == "failed" and self.failure_code is None:
            raise ValueError("failed window attempts require a failure code")
        if self.status != "failed" and self.failure_code is not None:
            raise ValueError("only failed window attempts may carry a failure code")
        return self


class CoverageProof(FrozenContract):
    asset_id: NonEmptyStr
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    window_intervals: tuple[tuple[int, int], ...]
    covered_frames: tuple[int, ...]
    missing_frames: tuple[int, ...]

    @model_validator(mode="after")
    def is_an_exact_coverage_accounting(self) -> CoverageProof:
        if self.end_frame <= self.start_frame:
            raise ValueError("coverage interval must be a non-empty half-open interval")
        required = tuple(range(self.start_frame, self.end_frame))
        expected_covered = tuple(
            frame
            for frame in required
            if any(start <= frame < end for start, end in self.window_intervals)
        )
        if self.covered_frames != expected_covered:
            raise ValueError("coverage proof does not match its window intervals")
        expected_missing = tuple(frame for frame in required if frame not in self.covered_frames)
        if self.missing_frames != expected_missing:
            raise ValueError("coverage proof does not account for every required frame")
        return self

    @computed_field(return_type=Sha256)
    @property
    def proof_hash(self) -> str:
        return _canonical_hash(self.model_dump(exclude={"proof_hash"}))


class WindowPlan(FrozenContract):
    asset_id: NonEmptyStr
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    attempts: tuple[WindowAttempt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def attempts_are_confined_and_unambiguous(self) -> WindowPlan:
        if self.end_frame <= self.start_frame:
            raise ValueError("plan interval must be a non-empty half-open interval")
        ordinals: set[int] = set()
        for attempt in self.attempts:
            window = attempt.window
            if window.asset_id != self.asset_id:
                raise ValueError("attempt must reference the plan asset")
            if not self.start_frame <= window.start_frame < window.end_frame <= self.end_frame:
                raise ValueError("attempt must fall inside the plan interval")
            if window.ordinal in ordinals:
                raise ValueError("window attempt ordinals must be unique")
            ordinals.add(window.ordinal)
        return self

    @property
    def windows(self) -> tuple[PropagationWindow, ...]:
        """Final active windows, excluding failed and superseded historical attempts."""

        return tuple(
            attempt.window
            for attempt in self.attempts
            if attempt.status in {"committed", "planned"}
        )

    @property
    def coverage_proof(self) -> CoverageProof:
        intervals = tuple((window.start_frame, window.end_frame) for window in self.windows)
        required = tuple(range(self.start_frame, self.end_frame))
        covered = tuple(
            frame for frame in required if any(start <= frame < end for start, end in intervals)
        )
        return CoverageProof(
            asset_id=self.asset_id,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            window_intervals=intervals,
            covered_frames=covered,
            missing_frames=tuple(frame for frame in required if frame not in covered),
        )

    @computed_field(return_type=Sha256)
    @property
    def plan_hash(self) -> str:
        return _canonical_hash(self.model_dump(exclude={"plan_hash"}))

    def with_attempt_status(
        self,
        *,
        ordinal: int,
        status: Literal["planned", "committed", "failed", "superseded"],
        failure_code: str | None = None,
    ) -> WindowPlan:
        """Return a new plan with one explicit attempt lifecycle transition."""

        replaced = False
        attempts: list[WindowAttempt] = []
        for attempt in self.attempts:
            if attempt.window.ordinal == ordinal:
                replaced = True
                attempts.append(
                    WindowAttempt(window=attempt.window, status=status, failure_code=failure_code)
                )
            else:
                attempts.append(attempt)
        if not replaced:
            raise ValueError("window attempt ordinal is not in this plan")
        return WindowPlan(
            asset_id=self.asset_id,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            attempts=tuple(attempts),
        )


class RelayEvidence(FrozenContract):
    predecessor_window: PropagationWindow
    successor_window: PropagationWindow
    instance_id: NonEmptyStr
    class_name: NonEmptyStr
    prompt_frame_index: int = Field(ge=0)
    relay_frame: int = Field(ge=0)
    relay_mask_ref: str | None = None
    absence: InstanceAbsence | None = None
    agreement_frames: tuple[int, ...]

    @field_validator("relay_mask_ref")
    @classmethod
    def relay_mask_is_confined(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("relay mask reference must be relative and confined")
        return value

    @model_validator(mode="after")
    def is_exact_start_relay_evidence(self) -> RelayEvidence:
        if self.predecessor_window.asset_id != self.successor_window.asset_id:
            raise ValueError("relay windows must reference the same asset")
        if self.relay_frame != self.successor_window.start_frame:
            raise ValueError("relay frame must be the successor start frame")
        if not (
            self.predecessor_window.start_frame
            <= self.relay_frame
            < self.predecessor_window.end_frame
        ):
            raise ValueError("relay frame must be present in the predecessor window")
        if not self.agreement_frames:
            raise ValueError("relay link requires a shared non-relay agreement frame")
        if self.agreement_frames != agreement_frames(
            self.predecessor_window, self.successor_window
        ):
            raise ValueError("relay agreement frames must be the exact non-relay intersection")
        if (self.relay_mask_ref is None) == (self.absence is None):
            raise ValueError("relay evidence requires exactly one mask or explicit absence")
        if self.absence is not None and (
            self.absence.asset_id != self.successor_window.asset_id
            or self.absence.instance_id != self.instance_id
            or self.absence.class_name != self.class_name
            or self.absence.frame_index != self.relay_frame
            or self.absence.prompt_frame_index != self.prompt_frame_index
        ):
            raise ValueError("relay absence must describe the exact relay instance and frame")
        return self


def initial_window_plan(
    *,
    asset_id: str,
    start_frame: int,
    end_frame: int,
    window_size: int = 30,
    overlap: int = 10,
) -> WindowPlan:
    windows = plan_windows(
        asset_id=asset_id,
        start_frame=start_frame,
        end_frame=end_frame,
        window_size=window_size,
        overlap=overlap,
    )
    return WindowPlan(
        asset_id=asset_id,
        start_frame=start_frame,
        end_frame=end_frame,
        attempts=tuple(WindowAttempt(window=window) for window in windows),
    )


def replan_after_oom(
    *,
    plan: WindowPlan,
    failed_window: PropagationWindow | WindowAttempt,
    window_size: int = 15,
    overlap: int = 5,
) -> WindowPlan:
    failed = failed_window.window if isinstance(failed_window, WindowAttempt) else failed_window
    failed_ordinal = failed.ordinal
    matching_attempts = [attempt for attempt in plan.attempts if attempt.window == failed]
    if len(matching_attempts) != 1:
        raise ValueError("failed window must be an attempt in the current plan")
    original = matching_attempts[0]
    if original.status in {"committed", "superseded"}:
        raise ValueError("only an active uncommitted window can fail with CUDA OOM")

    attempts: list[WindowAttempt] = []
    for attempt in plan.attempts:
        if attempt.window.ordinal == failed_ordinal:
            attempts.append(
                WindowAttempt(window=attempt.window, status="failed", failure_code="cuda_oom")
            )
        elif attempt.window.start_frame >= failed.start_frame and attempt.status == "planned":
            attempts.append(WindowAttempt(window=attempt.window, status="superseded"))
        elif attempt.window.start_frame >= failed.start_frame and attempt.status == "committed":
            raise ValueError("cannot discard committed evidence after a failed window")
        else:
            attempts.append(attempt)

    replacement = plan_windows(
        asset_id=plan.asset_id,
        start_frame=failed.start_frame,
        end_frame=plan.end_frame,
        window_size=window_size,
        overlap=overlap,
    )
    next_ordinal = max(attempt.window.ordinal for attempt in attempts) + 1
    attempts.extend(
        WindowAttempt(
            window=window.model_copy(update={"ordinal": next_ordinal + index}), status="planned"
        )
        for index, window in enumerate(replacement)
    )
    return WindowPlan(
        asset_id=plan.asset_id,
        start_frame=plan.start_frame,
        end_frame=plan.end_frame,
        attempts=tuple(attempts),
    )


def agreement_frames(
    predecessor: PropagationWindow,
    successor: PropagationWindow,
) -> tuple[int, ...]:
    if predecessor.asset_id != successor.asset_id:
        raise ValueError("agreement windows must reference the same asset")
    start = max(predecessor.start_frame, successor.start_frame)
    end = min(predecessor.end_frame, successor.end_frame)
    return tuple(frame for frame in range(start, max(start, end)) if frame != successor.start_frame)


def relay_successor(
    *,
    predecessor: PropagationResult,
    successor: PropagationWindow,
) -> tuple[RelayEvidence, ...]:
    """Build successor prompts from masks at exactly its first source frame."""

    if predecessor.window.asset_id != successor.asset_id:
        raise ValueError("relay windows must reference the same asset")
    relay_frame = successor.start_frame
    if not predecessor.window.start_frame <= relay_frame < predecessor.window.end_frame:
        raise ValueError("successor start frame is not available from the predecessor")
    shared = agreement_frames(predecessor.window, successor)
    observations = {
        observation.instance_id: observation
        for observation in predecessor.observations
        if observation.frame_index == relay_frame
    }
    evidence: list[RelayEvidence] = []
    for prompt in sorted(predecessor.prompts, key=lambda item: item.instance_id):
        observation = observations.get(prompt.instance_id)
        absence = None
        mask_ref = None
        if observation is None:
            absence = InstanceAbsence(
                asset_id=successor.asset_id,
                frame_index=relay_frame,
                instance_id=prompt.instance_id,
                class_name=prompt.class_name,
                prompt_frame_index=prompt.frame_index,
                reason="missing_relay_mask",
            )
        else:
            mask_ref = observation.mask_ref
        evidence.append(
            RelayEvidence(
                predecessor_window=predecessor.window,
                successor_window=successor,
                instance_id=prompt.instance_id,
                class_name=prompt.class_name,
                prompt_frame_index=prompt.frame_index,
                relay_frame=relay_frame,
                relay_mask_ref=mask_ref,
                absence=absence,
                agreement_frames=shared,
            )
        )
    return tuple(evidence)
