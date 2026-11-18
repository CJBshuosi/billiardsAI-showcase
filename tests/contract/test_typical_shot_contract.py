"""Public contract tests for deterministic typical-shot selection."""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import (
    TypicalShotRuleVersion,
    TypicalShotSelection,
    TypicalShotSelectionReason,
    TypicalShotSelectionResult,
)


def _selection(
    rank: int,
    reason: TypicalShotSelectionReason,
) -> TypicalShotSelection:
    return TypicalShotSelection(
        shot_id=UUID(f"00000000-0000-4000-8000-{rank:012d}"),
        sequence=rank,
        selection_rank=rank,
        selection_reason=reason,
        rule_version=TypicalShotRuleVersion.V1,
    )


def _result(*selections: TypicalShotSelection) -> TypicalShotSelectionResult:
    return TypicalShotSelectionResult(
        session_id=UUID("10000000-0000-4000-8000-000000000001"),
        report_id=UUID("20000000-0000-4000-8000-000000000001"),
        report_revision=1,
        rule_version=TypicalShotRuleVersion.V1,
        selections=selections,
    )


def test_typical_shot_contract_is_strict_frozen_and_rejects_python_extras():
    selection = _selection(1, TypicalShotSelectionReason.SEVERE_FOUL)
    result = _result(selection)

    with pytest.raises(ValidationError, match="frozen"):
        result.report_revision = 2
    with pytest.raises(ValidationError, match="report_revision"):
        TypicalShotSelectionResult.model_validate(result.model_dump() | {"report_revision": True})
    with pytest.raises(ValidationError, match="unexpected"):
        TypicalShotSelection.model_validate(selection.model_dump() | {"unexpected": "raw"})


def test_typical_shot_contract_accepts_typed_json_and_rejects_json_extras():
    result = _result(_selection(1, TypicalShotSelectionReason.POSITIVE_POTTING))
    payload = result.model_dump(mode="json")

    reparsed = TypicalShotSelectionResult.model_validate_json(json.dumps(payload))

    assert reparsed == result
    payload["media_url"] = "https://example.invalid/shot.mp4"
    with pytest.raises(ValidationError, match="media_url"):
        TypicalShotSelectionResult.model_validate_json(json.dumps(payload))


def test_result_enforces_rank_identity_reason_version_and_category_quotas():
    severe = _selection(1, TypicalShotSelectionReason.SEVERE_FOUL)
    repeated = _selection(2, TypicalShotSelectionReason.REPEATED_MISSED_POT)
    positive = _selection(3, TypicalShotSelectionReason.POSITIVE_POTTING)

    invalid_sets = (
        (severe, severe.model_copy(update={"selection_rank": 2})),
        (severe, severe.model_copy(update={"shot_id": repeated.shot_id})),
        (severe, severe.model_copy(update={"selection_rank": 3})),
        (
            severe,
            TypicalShotSelection.model_construct(
                **(severe.model_dump() | {"rule_version": "future"})
            ),
        ),
        (
            repeated,
            repeated.model_copy(
                update={"shot_id": severe.shot_id, "sequence": 1, "selection_rank": 2}
            ),
        ),
        (
            positive,
            positive.model_copy(
                update={"shot_id": repeated.shot_id, "sequence": 2, "selection_rank": 2}
            ),
        ),
    )
    for selections in invalid_sets:
        with pytest.raises(ValidationError):
            _result(*selections)


@pytest.mark.parametrize(
    "update",
    [
        {"raw_observation": {"provider": "must-not-leak"}},
        {"report_revision": True},
        {"rule_version": "future"},
        {
            "selections": tuple(
                _selection(rank, reason)
                for rank, reason in enumerate(
                    (
                        TypicalShotSelectionReason.SEVERE_FOUL,
                        TypicalShotSelectionReason.SEVERE_DEVIATION,
                        TypicalShotSelectionReason.REPEATED_FOUL,
                        TypicalShotSelectionReason.REPEATED_LARGE_DEVIATION,
                        TypicalShotSelectionReason.POSITIVE_POTTING,
                        TypicalShotSelectionReason.SEVERE_MISSED_POT,
                    ),
                    start=1,
                )
            )
        },
    ],
)
def test_public_result_model_copy_cannot_bypass_strict_validation(update):
    result = _result(_selection(1, TypicalShotSelectionReason.SEVERE_FOUL))

    with pytest.raises(ValidationError):
        result.model_copy(update=update)


def test_result_allows_zero_to_five_items_but_never_six():
    assert _result().selections == ()
    selections = (
        _selection(1, TypicalShotSelectionReason.SEVERE_FOUL),
        _selection(2, TypicalShotSelectionReason.SEVERE_DEVIATION),
        _selection(3, TypicalShotSelectionReason.REPEATED_FOUL),
        _selection(4, TypicalShotSelectionReason.REPEATED_MISSED_POT),
        _selection(5, TypicalShotSelectionReason.POSITIVE_POTTING),
    )
    assert len(_result(*selections).selections) == 5
    sixth = _selection(6, TypicalShotSelectionReason.SEVERE_MISSED_POT)
    with pytest.raises(ValidationError):
        _result(*selections, sixth)


def test_public_schema_and_serialization_never_expose_raw_media_or_urls():
    result = _result(_selection(1, TypicalShotSelectionReason.SEVERE_DEVIATION))
    forbidden = {
        "raw_observation",
        "trajectory",
        "media",
        "media_locator",
        "media_url",
        "image_url",
        "video_url",
    }
    schema = json.dumps(TypicalShotSelectionResult.model_json_schema()).lower()
    serialized = json.dumps(result.model_dump(mode="json")).lower()

    assert all(name not in schema for name in forbidden)
    assert all(name not in serialized for name in forbidden)
    assert TypicalShotSelectionResult.model_config["strict"] is True
    assert TypicalShotSelectionResult.model_config["frozen"] is True
    assert TypicalShotSelectionResult.model_config["extra"] == "forbid"


def test_nested_unsafe_extra_is_rejected_when_building_public_result():
    selection = _selection(1, TypicalShotSelectionReason.SEVERE_FOUL)
    tampered = TypicalShotSelection.model_construct(**selection.model_dump())
    vars(tampered)["raw_observation"] = {"provider": "must-not-leak"}

    with pytest.raises(ValidationError, match="raw_observation"):
        _result(tampered)
