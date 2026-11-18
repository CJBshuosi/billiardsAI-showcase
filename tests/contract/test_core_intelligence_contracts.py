from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.coaching import CoachAssessment
from billiards_ai.shared.models.events import DomainEvent
from billiards_ai.shared.models.evidence import ApprovedEvidenceBatch, ShotEvidenceEnvelope
from billiards_ai.shared.models.growth import AbilityMetric, GrowthEvidenceSnapshot, PlanOutcome
from billiards_ai.shared.models.realtime import RealtimeShotObservation
from billiards_ai.shared.models.training_plan import PublishedPlanBundle, TrainingPlan

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "core_intelligence" / "v1"
FIXTURE_MODELS = {
    "shot-evidence-envelope.json": ShotEvidenceEnvelope,
    "approved-evidence-batch.json": ApprovedEvidenceBatch,
    "growth-evidence-snapshot.json": GrowthEvidenceSnapshot,
    "coach-assessment.json": CoachAssessment,
    "training-plan.json": TrainingPlan,
    "published-plan-bundle.json": PublishedPlanBundle,
    "realtime-shot-observation.json": RealtimeShotObservation,
    "plan-outcome.json": PlanOutcome,
}
FORBIDDEN_KEYS = {
    "openid",
    "session_key",
    "app_secret",
    "access_token",
    "refresh_token",
    "prompt",
    "prompt_text",
    "raw_observation",
    "video_url",
    "media_url",
    "media_path",
}


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _parse(name: str, model: type):
    return model.model_validate_json((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | {
            nested for item in value.values() for nested in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _all_keys(item)}
    return set()


@pytest.mark.parametrize(("filename", "model"), FIXTURE_MODELS.items())
def test_frozen_v1_fixtures_round_trip(filename: str, model: type) -> None:
    payload = _fixture(filename)

    parsed = _parse(filename, model)
    reparsed = model.model_validate_json(parsed.model_dump_json())

    assert reparsed == parsed
    assert parsed.schema_version == "1.0"
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps({**payload, "unexpected": "forbidden"}))
    with pytest.raises(ValidationError):
        parsed.__setattr__("schema_version", "2.0")


def test_fixture_chain_preserves_identity_and_lineage() -> None:
    batch = _parse("approved-evidence-batch.json", ApprovedEvidenceBatch)
    snapshot = _parse("growth-evidence-snapshot.json", GrowthEvidenceSnapshot)
    assessment = _parse("coach-assessment.json", CoachAssessment)
    plan = _parse("training-plan.json", TrainingPlan)
    bundle = _parse("published-plan-bundle.json", PublishedPlanBundle)
    observation = _parse("realtime-shot-observation.json", RealtimeShotObservation)
    outcome = _parse("plan-outcome.json", PlanOutcome)

    assert batch.user_id == snapshot.user_id == assessment.user_id == plan.user_id
    assert batch.batch_id in snapshot.source_batch_ids
    assert assessment.snapshot_id == snapshot.snapshot_id
    assert assessment.snapshot_version == snapshot.version
    assert plan.assessment_id == assessment.assessment_id
    assert bundle.plan.plan_id == plan.plan_id
    assert bundle.plan.version == plan.version
    assert observation.session_id == outcome.session_id
    assert outcome.plan_id == plan.plan_id
    assert outcome.direct_ability_update_allowed is False
    envelope = _parse("shot-evidence-envelope.json", ShotEvidenceEnvelope)
    with pytest.raises(ValidationError):
        envelope.shot.__setattr__("sequence", 2)


def test_assessment_rejects_evidence_refs_absent_from_snapshot_context() -> None:
    payload = _fixture("coach-assessment.json")
    payload["findings"][0]["evidence_ref_ids"] = ["00000000-0000-4000-8000-000000000999"]

    with pytest.raises(ValidationError, match="snapshot evidence"):
        CoachAssessment.model_validate_json(json.dumps(payload))


def test_training_plan_rejects_non_contiguous_blocks_and_wrong_assessment() -> None:
    payload = _fixture("training-plan.json")
    payload["blocks"][0]["order"] = 2

    with pytest.raises(ValidationError, match="contiguous"):
        TrainingPlan.model_validate_json(json.dumps(payload))


def test_published_bundle_rejects_tampered_embedded_plan() -> None:
    payload = _fixture("published-plan-bundle.json")
    payload["plan"]["version"] = 2

    with pytest.raises(ValidationError, match="plan hash"):
        PublishedPlanBundle.model_validate_json(json.dumps(payload))


def test_realtime_observation_rejects_cross_session_shot() -> None:
    payload = _fixture("realtime-shot-observation.json")
    payload["shot"]["session_id"] = "00000000-0000-4000-8000-000000000888"

    with pytest.raises(ValidationError, match="session"):
        RealtimeShotObservation.model_validate_json(json.dumps(payload))


def test_plan_outcome_cannot_authorize_direct_ability_update() -> None:
    payload = _fixture("plan-outcome.json")
    payload["direct_ability_update_allowed"] = True

    with pytest.raises(ValidationError):
        PlanOutcome.model_validate_json(json.dumps(payload))


def test_insufficient_metric_preserves_sample_count_without_inventing_a_score() -> None:
    payload = {
        "metric_code": "rail_shot.success_rate",
        "taxonomy": {"skill_code": "POTTING", "rail_context": "FROZEN_TO_RAIL"},
        "support_status": "INSUFFICIENT_EVIDENCE",
        "sample_count": 2,
        "coverage": 0.05,
        "reliability": "LOW",
        "success_rate": None,
        "error_p50_mm": None,
        "error_p90_mm": None,
        "stability": None,
    }

    metric = AbilityMetric.model_validate_json(json.dumps(payload))

    assert metric.sample_count == 2
    assert metric.success_rate is None


def test_domain_event_requires_v1_name_positive_version_and_utc_time() -> None:
    now = datetime.now(UTC)
    valid = {
        "schema_version": "1.0",
        "event_id": "00000000-0000-4000-8000-000000000100",
        "event_type": "growth.snapshot.created.v1",
        "aggregate_type": "growth_snapshot",
        "aggregate_id": "00000000-0000-4000-8000-000000000101",
        "aggregate_version": 1,
        "occurred_at": now.isoformat(),
        "causation_id": None,
        "correlation_id": "00000000-0000-4000-8000-000000000102",
        "idempotency_key": "snapshot:101:1",
        "payload": {"snapshot_version": 1},
    }

    assert DomainEvent.model_validate_json(json.dumps(valid)).event_type.endswith(".v1")
    for field, value in (
        ("event_type", "growth.snapshot.created"),
        ("aggregate_version", 0),
        ("occurred_at", (now.replace(tzinfo=None) + timedelta(seconds=1)).isoformat()),
    ):
        invalid = {**valid, field: value}
        with pytest.raises(ValidationError):
            DomainEvent.model_validate_json(json.dumps(invalid))


def test_fixtures_exclude_sensitive_and_raw_fields() -> None:
    for filename in FIXTURE_MODELS:
        keys = _all_keys(_fixture(filename))
        assert keys.isdisjoint(FORBIDDEN_KEYS), f"{filename}: {keys & FORBIDDEN_KEYS}"


def test_production_contract_modules_do_not_import_lab_or_provider_sdks() -> None:
    contract_files = [
        Path("src/billiards_ai/shared/models/events.py"),
        Path("src/billiards_ai/shared/models/evidence.py"),
        Path("src/billiards_ai/shared/models/growth.py"),
        Path("src/billiards_ai/shared/models/coaching.py"),
        Path("src/billiards_ai/shared/models/training_plan.py"),
        Path("src/billiards_ai/shared/models/realtime.py"),
    ]
    forbidden = ("billiards_ai.lab", "openai", "anthropic", "deepseek", "kimi")

    for path in contract_files:
        source = path.read_text(encoding="utf-8").lower()
        assert all(item not in source for item in forbidden), path
