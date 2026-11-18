"""Story 4.1 public review and approved-data contracts."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import (
    ApprovedSessionData,
    ApprovedShotData,
    ReportDeliveryDecision,
    ReportNotDeliverableReason,
    ReviewTask,
    ReviewTaskStatus,
)


def test_review_task_is_strict_frozen_and_forbids_extra_in_python_mode():
    task = ReviewTask(
        id=uuid4(),
        session_id=uuid4(),
        report_id=uuid4(),
        report_revision=1,
        status=ReviewTaskStatus.APPROVED,
    )

    with pytest.raises(ValidationError, match="frozen"):
        task.status = ReviewTaskStatus.REJECTED
    with pytest.raises(ValidationError, match="report_revision"):
        ReviewTask.model_validate(task.model_dump() | {"report_revision": "1"})
    with pytest.raises(ValidationError, match="unexpected"):
        ReviewTask.model_validate(task.model_dump() | {"unexpected": True})


def test_review_task_accepts_typed_json_without_weakening_python_validation():
    payload = {
        "id": str(uuid4()),
        "session_id": str(uuid4()),
        "report_id": str(uuid4()),
        "report_revision": 2,
        "status": "corrected",
    }

    task = ReviewTask.model_validate_json(json.dumps(payload))

    assert task.status is ReviewTaskStatus.CORRECTED
    assert task.report_revision == 2


@pytest.mark.parametrize(
    ("deliverable", "reason"),
    [
        (True, ReportNotDeliverableReason.REVIEW_PENDING),
        (False, None),
    ],
)
def test_delivery_decision_cannot_contradict_its_reason(deliverable, reason):
    with pytest.raises(ValidationError, match="reason"):
        ReportDeliveryDecision(deliverable=deliverable, reason=reason)


def test_approved_projection_schema_has_no_raw_observation_or_media_locator():
    forbidden = {"raw_observation", "trajectory", "media_locator", "media_url", "raw_video_url"}
    schema = ApprovedSessionData.model_json_schema()
    serialized = json.dumps(schema)

    assert all(name not in serialized for name in forbidden)
    assert schema["additionalProperties"] is False
    assert ApprovedShotData.model_config["strict"] is True
    assert ApprovedShotData.model_config["frozen"] is True
    assert ApprovedShotData.model_config["extra"] == "forbid"
