from billiards_ai.lab.render import build_overlay_frames
from billiards_ai.lab.report import build_report_html
from billiards_ai.lab.runner import RunInputs


def test_every_displayed_event_links_to_evidence_frame() -> None:
    inputs = RunInputs(
        dataset_revision="dataset-r1",
        candidate_id="candidate-a",
        manifest={"seed": 7},
        predictions=(
            {
                "shot_id": "shot-1",
                "events": [
                    {
                        "event_id": "pot-1",
                        "type": "pot",
                        "timestamp_ms": 500,
                        "supported": True,
                        "confidence": 0.8,
                        "evidence_frame_ids": [12, 13],
                    }
                ],
            },
        ),
        metrics={},
        failure_records=(),
        evidence_frames={
            12: {"timestamp_ms": 480, "labels": ["8"]},
            13: {"timestamp_ms": 520, "labels": ["pocket"]},
        },
    )

    frames = build_overlay_frames(inputs.predictions, inputs.evidence_frames)
    report = build_report_html(inputs.predictions, inputs.metrics)

    assert frames[0]["timestamp_ms"] == 480
    assert frames[0]["events"][0]["evidence_frame_ids"] == [12, 13]
    assert 'id="evidence-frame-12"' in report
    assert 'href="#evidence-frame-12"' in report


def test_unsupported_field_is_not_rendered_as_zero() -> None:
    predictions = (
        {
            "shot_id": "shot-unsupported",
            "events": [
                {
                    "event_id": "foul-unknown",
                    "type": "foul",
                    "timestamp_ms": 900,
                    "supported": False,
                    "confidence": None,
                    "evidence_frame_ids": [22],
                    "abstention_reason": "ruleset_not_configured",
                }
            ],
        },
    )

    report = build_report_html(predictions, {"foul.accuracy": None})

    assert "unsupported" in report
    assert "ruleset_not_configured" in report
    assert "foul.accuracy: 0" not in report
