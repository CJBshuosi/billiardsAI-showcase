"""Frozen and versioned public report contract for Story 3.3."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.quality import SessionQualityStatus
from billiards_ai.shared.models.report import (
    ReportDocument,
    ReportHeadline,
    ReportHeadlineCode,
    ReportStatus,
    UnsupportedReportSchemaVersionError,
    parse_report_document,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "reports" / "v1"
FORBIDDEN_KEYS = {
    "raw_observation",
    "media_locator",
    "media_url",
    "raw_video_url",
    "video_url",
}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _all_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(item for child in value.values() for item in _all_strings(child))
    if isinstance(value, list):
        return tuple(item for child in value for item in _all_strings(child))
    return (value,) if isinstance(value, str) else ()


@pytest.mark.parametrize(
    "fixture_name",
    ["normal.json", "partial-abstain.json", "unscorable.json"],
)
def test_golden_report_fixtures_validate_as_v1_without_private_media_fields(fixture_name: str):
    payload = json.loads((FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8"))

    document = parse_report_document(payload)

    assert document.schema_version == "1.0"
    assert document.report_status is ReportStatus.DRAFT_PENDING_REVIEW
    assert document.typical_shots == ()
    assert document.coaching_recommendations == ()
    assert FORBIDDEN_KEYS.isdisjoint(_all_keys(payload))
    assert not any(value.startswith(("http://", "https://")) for value in _all_strings(payload))


def test_json_schema_is_strict_and_excludes_private_or_delivery_fields():
    schema = ReportDocument.model_json_schema()

    assert schema["additionalProperties"] is False
    assert FORBIDDEN_KEYS.isdisjoint(_all_keys(schema))
    assert {"deliverable", "approved"}.isdisjoint(schema["properties"])


def test_report_contract_is_frozen_and_revision_parent_is_consistent():
    payload = json.loads((FIXTURE_ROOT / "normal.json").read_text(encoding="utf-8"))
    document = parse_report_document(payload)

    with pytest.raises(ValidationError, match="frozen"):
        document.revision = 2
    with pytest.raises(ValidationError, match="first revision"):
        parse_report_document(
            payload | {"parent_revision_id": "30000000-0000-4000-8000-000000000099"}
        )
    with pytest.raises(ValidationError, match="later revision"):
        parse_report_document(payload | {"revision": 2})


def test_report_contract_rejects_python_mode_numeric_string_coercion():
    payload = json.loads((FIXTURE_ROOT / "normal.json").read_text(encoding="utf-8"))
    python_payload = parse_report_document(payload).model_dump(mode="python")
    python_payload["session_statistics"]["pot_rate"] = "0.5"

    with pytest.raises(ValidationError, match="session_statistics"):
        ReportDocument.model_validate(python_payload)


def test_unsupported_schema_major_version_raises_typed_error():
    payload = json.loads((FIXTURE_ROOT / "normal.json").read_text(encoding="utf-8"))

    with pytest.raises(UnsupportedReportSchemaVersionError) as raised:
        parse_report_document(payload | {"schema_version": "2.0"})

    assert raised.value.schema_version == "2.0"
    assert raised.value.supported_major_versions == (1,)


def test_report_rejects_raw_observations_and_video_urls_at_the_contract_boundary():
    payload = json.loads((FIXTURE_ROOT / "normal.json").read_text(encoding="utf-8"))

    with pytest.raises(ValidationError, match="raw_observation"):
        parse_report_document(payload | {"raw_observation": {"provider": "private"}})
    with pytest.raises(ValidationError, match="URLs are not allowed"):
        parse_report_document(
            payload
            | {
                "typical_shots": [
                    {
                        "shot_id": "50000000-0000-4000-8000-000000000001",
                        "summary": "原始视频 https://camera.example.test/raw.mp4",
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "fixture_name",
    ["complete.json", "partial-content.json", "no-reliable-content.json"],
)
def test_v11_golden_reports_are_strict_round_trippable_and_complete(fixture_name: str):
    fixture = FIXTURE_ROOT / "v1.1" / fixture_name
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    document = parse_report_document(payload)

    assert document.schema_version == "1.1"
    assert document.model_dump(mode="json") == payload
    assert document.typical_shots == ()
    assert document.coaching_recommendations == ()
    assert document.headline is not None
    assert document.trust_scope is not None
    assert document.objective_statistics is not None
    assert document.typical_shot_analysis is not None
    assert document.training_recommendation is not None
    assert document.quality_notice is not None
    assert len(document.content_sha256) == 64


def test_schema_minor_matrix_prevents_dual_truth_and_unknown_versions():
    legacy = json.loads((FIXTURE_ROOT / "normal.json").read_text(encoding="utf-8"))
    complete = json.loads((FIXTURE_ROOT / "v1.1" / "complete.json").read_text(encoding="utf-8"))

    with pytest.raises(ValidationError):
        parse_report_document(legacy | {"schema_version": "1.1"})
    with pytest.raises(ValidationError):
        parse_report_document(
            complete
            | {
                "typical_shots": [
                    {
                        "shot_id": "50000000-0000-4000-8000-000000000001",
                        "summary": "旧字段不得形成第二套真相",
                    }
                ]
            }
        )
    with pytest.raises(ValidationError):
        parse_report_document(complete | {"schema_version": "1.0"})
    with pytest.raises(UnsupportedReportSchemaVersionError):
        parse_report_document(legacy | {"schema_version": "1.2"})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("revision",), True),
        (("objective_statistics", "source", "report_revision"), "1"),
        (("headline", "text"), "恶意\n控制字符"),
        (("headline", "text"), "https://example.test/report"),
        (("headline", "text"), "/private/report.json"),
    ],
)
def test_v11_rejects_bool_coercion_and_unsafe_text(path, value):
    payload = json.loads(
        (FIXTURE_ROOT / "v1.1" / "no-reliable-content.json").read_text(encoding="utf-8")
    )
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        parse_report_document(payload)


def test_v11_recursively_revalidates_constructed_and_copied_nested_models():
    payload = json.loads((FIXTURE_ROOT / "v1.1" / "complete.json").read_text(encoding="utf-8"))
    report = parse_report_document(payload)
    malicious_headline = report.headline.model_construct(
        **(report.headline.model_dump(mode="python") | {"text": "oss://private/report"}),
    )

    malicious_report = report.model_copy(update={"headline": malicious_headline})
    with pytest.raises(ValidationError):
        ReportDocument.model_validate(dict(vars(malicious_report)))


@pytest.mark.parametrize(
    "mutation",
    ["headline", "coverage", "notice_boundaries", "trust_boundaries"],
)
def test_v11_top_level_derived_fields_must_match_authoritative_sections(mutation):
    report = parse_report_document(
        json.loads((FIXTURE_ROOT / "v1.1" / "complete.json").read_text(encoding="utf-8"))
    )
    values = report.model_dump(mode="python", exclude={"content_sha256"})
    if mutation == "headline":
        values["headline"] = ReportHeadline(code=ReportHeadlineCode.PARTIAL)
    elif mutation == "coverage":
        values["quality_notice"] = report.quality_notice.model_copy(
            update={"data_coverage_status": SessionQualityStatus.LOW_COVERAGE}
        )
    elif mutation == "notice_boundaries":
        values["quality_notice"] = report.quality_notice.model_copy(
            update={"algorithm_boundaries": ()}
        )
    else:
        values["trust_scope"] = report.trust_scope.model_copy(update={"unsupported_dimensions": ()})

    with pytest.raises(ValidationError):
        ReportDocument.build_v11(**values)
