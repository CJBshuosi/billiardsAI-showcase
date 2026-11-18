"""Local, default-disabled orchestration for Story 2.8 ambiguity adjudication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID

from pydantic import ConfigDict, Field, StrictBool, ValidationError, model_validator

from billiards_ai.shared.models.access import ThirdPartyProcessingApproval
from billiards_ai.shared.models.adjudication import (
    AdjudicatedStruckBall,
    AdjudicationCandidate,
    AdjudicationRawObservation,
    AdjudicationRequest,
    AdjudicationResult,
    AdjudicationSource,
    AdjudicationStatus,
)
from billiards_ai.shared.models.observation import (
    ObjectiveShotObservation,
    ObservationConfidence,
)
from billiards_ai.shared.models.privacy import MaterialUsePurpose, StrictFrozenModel


class AdjudicationPolicy(StrictFrozenModel):
    """Local policy. Its empty default cannot identify or call a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    enabled: StrictBool = False
    provider_id: str | None = None
    endpoint_region: str | None = None
    purpose: MaterialUsePurpose | None = None
    cross_border: StrictBool | None = None
    estimated_cost_cny: Decimal = Field(default=Decimal("0"), ge=0, decimal_places=2)
    max_cost_per_shot_cny: Decimal = Field(default=Decimal("0"), ge=0, decimal_places=2)

    @model_validator(mode="after")
    def enabled_policy_is_complete(self) -> AdjudicationPolicy:
        if not self.enabled:
            return self
        if not self.provider_id or not self.endpoint_region or self.purpose is None:
            raise ValueError("enabled adjudication requires provider, region, and purpose")
        if self.cross_border is None:
            raise ValueError("enabled adjudication requires an explicit cross-border decision")
        if self.max_cost_per_shot_cny <= 0:
            raise ValueError("enabled adjudication requires a positive per-shot cost limit")
        return self


class ProviderApprovalReader(Protocol):
    def find_active_provider_approval(
        self, provider_id: str
    ) -> ThirdPartyProcessingApproval | None: ...


class AdjudicationProvider(Protocol):
    provider_id: str

    def adjudicate(self, request: AdjudicationRequest) -> str: ...


class ProviderAdjudicationError(RuntimeError):
    """A provider failure whose retry decision is safe for the finalizer to inspect."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class DisabledAdjudicationProvider:
    provider_id = "disabled"

    def adjudicate(self, request: AdjudicationRequest) -> str:
        raise ProviderAdjudicationError("adjudication provider is disabled", retryable=False)


class AdjudicationLedger(Protocol):
    """Atomic accounting boundary for one optional-provider attempt."""

    def claim(self, *, shot_id: UUID, signature: str, cost: Decimal) -> bool: ...

    def complete(
        self, *, shot_id: UUID, signature: str, result: AdjudicationResult, terminal: bool
    ) -> None: ...

    def cached(self, *, shot_id: UUID, signature: str) -> AdjudicationResult | None: ...

    def spent(self, *, shot_id: UUID) -> Decimal: ...


class InMemoryAdjudicationLedger:
    """Injectable, lock-protected ledger; production may replace it with durable storage."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._spent: dict[UUID, Decimal] = {}
        self._signatures: dict[UUID, str] = {}
        self._in_flight: set[UUID] = set()
        self._results: dict[UUID, AdjudicationResult] = {}

    def cached(self, *, shot_id: UUID, signature: str) -> AdjudicationResult | None:
        with self._lock:
            if self._signatures.get(shot_id) != signature:
                return None
            return self._results.get(shot_id)

    def spent(self, *, shot_id: UUID) -> Decimal:
        with self._lock:
            return self._spent.get(shot_id, Decimal("0"))

    def claim(self, *, shot_id: UUID, signature: str, cost: Decimal) -> bool:
        with self._lock:
            previous = self._signatures.get(shot_id)
            if previous is not None and (previous != signature or shot_id in self._in_flight):
                return False
            self._signatures[shot_id] = signature
            self._spent[shot_id] = self._spent.get(shot_id, Decimal("0")) + cost
            self._in_flight.add(shot_id)
            return True

    def complete(
        self, *, shot_id: UUID, signature: str, result: AdjudicationResult, terminal: bool
    ) -> None:
        with self._lock:
            if self._signatures.get(shot_id) != signature:
                return
            self._in_flight.discard(shot_id)
            if terminal:
                self._results[shot_id] = result


class JsonAdjudicationLedger:
    """Small durable ledger with an exclusive file lock for cross-process idempotency."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    def _read(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("adjudication ledger must be an object")
        return payload

    def _locked(self, operation):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read()
                result, changed = operation(state)
                if changed:
                    temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
                    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
                    os.replace(temporary, self.path)
                return result
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _entry(state: dict[str, dict[str, object]], shot_id: UUID) -> dict[str, object]:
        return state.setdefault(str(shot_id), {"spent": "0", "in_flight": False})

    def cached(self, *, shot_id: UUID, signature: str) -> AdjudicationResult | None:
        def operation(state):
            entry = state.get(str(shot_id), {})
            if entry.get("signature") != signature or "result" not in entry:
                return None, False
            return AdjudicationResult.model_validate(entry["result"]), False

        return self._locked(operation)

    def spent(self, *, shot_id: UUID) -> Decimal:
        def operation(state):
            return Decimal(str(state.get(str(shot_id), {}).get("spent", "0"))), False

        return self._locked(operation)

    def claim(self, *, shot_id: UUID, signature: str, cost: Decimal) -> bool:
        def operation(state):
            entry = self._entry(state, shot_id)
            previous = entry.get("signature")
            if previous is not None and (previous != signature or entry.get("in_flight") is True):
                return False, False
            entry["signature"] = signature
            entry["spent"] = str(Decimal(str(entry["spent"])) + cost)
            entry["in_flight"] = True
            return True, True

        return self._locked(operation)

    def complete(
        self, *, shot_id: UUID, signature: str, result: AdjudicationResult, terminal: bool
    ) -> None:
        def operation(state):
            entry = self._entry(state, shot_id)
            if entry.get("signature") != signature:
                return None, False
            entry["in_flight"] = False
            if terminal:
                entry["result"] = result.model_dump(mode="json")
            return None, True

        self._locked(operation)


class AmbiguityAdjudicator:
    """Adjudicate eligible observations without reading media or mutating a Shot."""

    def __init__(
        self,
        *,
        policy: AdjudicationPolicy | None = None,
        provider: AdjudicationProvider | None = None,
        approval_reader: ProviderApprovalReader | None = None,
        ledger: AdjudicationLedger | None = None,
    ):
        self.policy = policy or AdjudicationPolicy()
        self.provider = provider or DisabledAdjudicationProvider()
        self.approval_reader = approval_reader
        self.ledger = ledger or InMemoryAdjudicationLedger()

    @staticmethod
    def _idempotency_key(shot_id: UUID) -> str:
        return hashlib.sha256(f"adjudication:1.0:{shot_id}".encode()).hexdigest()

    @staticmethod
    def _candidates(observation: ObjectiveShotObservation) -> tuple[AdjudicationCandidate, ...]:
        candidates = {
            candidate.ball_id: AdjudicationCandidate(
                ball_id=candidate.ball_id,
                confidence=candidate.confidence,
                source=candidate.source,
            )
            for candidate in observation.struck_ball_candidates
            if candidate.confidence in {ObservationConfidence.LOW, ObservationConfidence.MEDIUM}
        }
        struck_ball = observation.struck_ball
        if struck_ball.value is not None and struck_ball.confidence in {
            ObservationConfidence.LOW,
            ObservationConfidence.MEDIUM,
        }:
            candidates[struck_ball.value] = AdjudicationCandidate(
                ball_id=struck_ball.value,
                confidence=struck_ball.confidence,
                source=struck_ball.source,
            )
        return tuple(candidates[ball_id] for ball_id in sorted(candidates))

    @staticmethod
    def _signature(candidates: tuple[AdjudicationCandidate, ...]) -> str:
        canonical = [candidate.model_dump(mode="json") for candidate in candidates]
        return hashlib.sha256(
            json.dumps(canonical, ensure_ascii=True, sort_keys=True).encode()
        ).hexdigest()

    @staticmethod
    def _result(
        *,
        shot_id: UUID,
        idempotency_key: str,
        evidence_signature: str,
        status: AdjudicationStatus,
        source: AdjudicationSource = AdjudicationSource.NONE,
        confidence: ObservationConfidence = ObservationConfidence.UNKNOWN,
        result: AdjudicatedStruckBall | None = None,
        retryable: bool = False,
        estimated_cost_cny: Decimal = Decimal("0"),
        provider_output: str | None = None,
        failure_detail: str | None = None,
    ) -> AdjudicationResult:
        return AdjudicationResult(
            shot_id=shot_id,
            idempotency_key=idempotency_key,
            evidence_signature=evidence_signature,
            result=result,
            source=source,
            confidence=confidence,
            status=status,
            retryable=retryable,
            estimated_cost_cny=estimated_cost_cny,
            raw_observation=AdjudicationRawObservation(
                provider_output=provider_output,
                failure_detail=failure_detail,
            ),
        )

    def _approval(self) -> ThirdPartyProcessingApproval | None:
        if self.approval_reader is None or self.policy.provider_id is None:
            return None
        try:
            return self.approval_reader.find_active_provider_approval(self.policy.provider_id)
        except Exception:
            return None

    def adjudicate(
        self, *, shot_id: UUID, observation: ObjectiveShotObservation
    ) -> AdjudicationResult:
        idempotency_key = self._idempotency_key(shot_id)
        candidates = self._candidates(observation)
        signature = self._signature(candidates)
        cached = self.ledger.cached(shot_id=shot_id, signature=signature)
        if cached is not None:
            return cached

        if observation.struck_ball.confidence is ObservationConfidence.HIGH or not candidates:
            result = self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.NOT_REQUIRED,
                source=AdjudicationSource.CLASSIC_CV,
                confidence=observation.struck_ball.confidence,
            )
            self.ledger.complete(shot_id=shot_id, signature=signature, result=result, terminal=True)
            return result

        if not self.policy.enabled:
            result = self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.DISABLED,
            )
            self.ledger.complete(shot_id=shot_id, signature=signature, result=result, terminal=True)
            return result

        projected_cost = self.ledger.spent(shot_id=shot_id) + self.policy.estimated_cost_cny
        if projected_cost > self.policy.max_cost_per_shot_cny:
            return self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.COST_LIMIT_EXCEEDED,
            )

        approval = self._approval()
        if approval is None or not approval.is_active:
            return self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.APPROVAL_REQUIRED,
            )
        if (
            approval.provider_id != self.policy.provider_id
            or approval.endpoint_region != self.policy.endpoint_region
            or approval.purpose is not self.policy.purpose
            or approval.cross_border is not self.policy.cross_border
            or self.provider.provider_id != self.policy.provider_id
        ):
            return self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.APPROVAL_MISMATCH,
            )
        if projected_cost > approval.max_cost_cny:
            return self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.COST_LIMIT_EXCEEDED,
            )

        request = AdjudicationRequest(
            shot_id=shot_id,
            idempotency_key=idempotency_key,
            evidence_signature=signature,
            candidates=candidates,
            estimated_cost_cny=self.policy.estimated_cost_cny,
        )
        if not self.ledger.claim(
            shot_id=shot_id, signature=signature, cost=self.policy.estimated_cost_cny
        ):
            return self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.IDEMPOTENCY_CONFLICT,
            )
        try:
            provider_output = self.provider.adjudicate(request)
        except ProviderAdjudicationError as exc:
            result = self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.PROVIDER_FAILED,
                source=AdjudicationSource.EXTERNAL_PROVIDER,
                retryable=exc.retryable,
                estimated_cost_cny=self.policy.estimated_cost_cny,
                failure_detail=str(exc),
            )
            self.ledger.complete(
                shot_id=shot_id, signature=signature, result=result, terminal=not exc.retryable
            )
            return result
        except Exception as exc:
            result = self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.PROVIDER_FAILED,
                source=AdjudicationSource.EXTERNAL_PROVIDER,
                estimated_cost_cny=self.policy.estimated_cost_cny,
                failure_detail=str(exc),
            )
            self.ledger.complete(shot_id=shot_id, signature=signature, result=result, terminal=True)
            return result

        if not isinstance(provider_output, str):
            result = self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.PARSE_FAILED,
                source=AdjudicationSource.EXTERNAL_PROVIDER,
                estimated_cost_cny=self.policy.estimated_cost_cny,
                failure_detail="provider returned non-text output",
            )
            self.ledger.complete(shot_id=shot_id, signature=signature, result=result, terminal=True)
            return result

        try:
            parsed = json.loads(provider_output)
            if not isinstance(parsed, dict):
                raise ValueError("provider response must be an object")
            ball_id = parsed["ball_id"]
            if isinstance(ball_id, bool) or not isinstance(ball_id, int):
                raise ValueError("provider ball ID must be an integer")
            if ball_id not in {candidate.ball_id for candidate in candidates}:
                raise ValueError("provider selected a candidate outside the request")
            confidence = ObservationConfidence(parsed["confidence"])
            if confidence is ObservationConfidence.UNKNOWN:
                raise ValueError("provider confidence cannot be unknown on success")
            adjudicated = AdjudicatedStruckBall(ball_id=ball_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            result = self._result(
                shot_id=shot_id,
                idempotency_key=idempotency_key,
                evidence_signature=signature,
                status=AdjudicationStatus.PARSE_FAILED,
                source=AdjudicationSource.EXTERNAL_PROVIDER,
                estimated_cost_cny=self.policy.estimated_cost_cny,
                provider_output=provider_output,
            )
            self.ledger.complete(shot_id=shot_id, signature=signature, result=result, terminal=True)
            return result

        result = self._result(
            shot_id=shot_id,
            idempotency_key=idempotency_key,
            evidence_signature=signature,
            status=AdjudicationStatus.SUCCEEDED,
            source=AdjudicationSource.EXTERNAL_PROVIDER,
            confidence=confidence,
            result=adjudicated,
            estimated_cost_cny=self.policy.estimated_cost_cny,
            provider_output=provider_output,
        )
        self.ledger.complete(shot_id=shot_id, signature=signature, result=result, terminal=True)
        return result
