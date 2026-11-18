import pytest
from pydantic import ValidationError

from billiards_ai.lab.propagation.evaluation import ReviewSample, ReviewUnit
from billiards_ai.lab.propagation.review_page import ReviewPageModel, render_review_page
from billiards_ai.lab.propagation.review_server import (
    ReviewVerdict,
    compute_verdict_summary,
)


def test_review_page_renders_one_escaped_frozen_unit_with_both_verdict_fields() -> None:
    page = render_review_page(
        ReviewPageModel(
            unit_id="a" * 64,
            unit_type="matched_pair",
            stratum="flagged_or_ambiguous_match",
            asset_id='<script>alert("asset")</script>',
            frame_index=42,
            media_url=f"/media/{'a' * 64}.jpg",
            evidence_lines=("匹配：ground-a → prediction-a", "QA：center_jump"),
            completed_count=3,
            total_count=10,
            revision=7,
        )
    )

    assert page.count("<form") == 1
    assert "第 4 / 10 项" in page
    assert "匹配：ground-a → prediction-a" in page
    assert "QA：center_jump" in page
    assert 'name="acceptable_without_redraw"' in page
    assert 'name="severe_drift_or_wrong_target"' in page
    assert 'name="revision" value="7"' in page
    assert 'name="unit_id" value="' + "a" * 64 + '"' in page
    assert "可直接使用，无需重画" in page
    assert "严重漂移或目标错误" in page
    assert "&lt;script&gt;" in page
    assert '<script>alert("asset")</script>' not in page
    assert "两个选项不能同时选择“是”" in page
    assert "漏检或误检不能标记为可直接使用" in page


def test_review_page_preserves_invalid_form_values_and_shows_localized_error() -> None:
    page = render_review_page(
        ReviewPageModel(
            unit_id="e" * 64,
            unit_type="matched_pair",
            stratum="clean_match",
            asset_id="video-a",
            frame_index=8,
            media_url=f"/media/{'e' * 64}.jpg",
            evidence_lines=("匹配证据",),
            completed_count=0,
            total_count=2,
            revision=4,
            acceptable_without_redraw=True,
            severe_drift_or_wrong_target=True,
            note="<b>保留这条备注</b>",
            validation_error="两个选项不能同时选择“是”。",
        )
    )

    assert "两个选项不能同时选择“是”。" in page
    assert page.count('value="true" checked') == 2
    assert "&lt;b&gt;保留这条备注&lt;/b&gt;" in page
    assert '<input type="hidden" name="revision" value="4">' in page


def test_verdict_summary_reports_distinct_raw_and_population_weighted_rates() -> None:
    units = (
        ReviewUnit(
            unit_id="1" * 64,
            unit_type="matched_pair",
            stratum="clean_match",
            asset_id="video-a",
            frame_index=1,
            ground_truth_id="ground-1",
            prediction_id="prediction-1",
        ),
        ReviewUnit(
            unit_id="2" * 64,
            unit_type="matched_pair",
            stratum="clean_match",
            asset_id="video-a",
            frame_index=2,
            ground_truth_id="ground-2",
            prediction_id="prediction-2",
        ),
        ReviewUnit(
            unit_id="3" * 64,
            unit_type="unmatched_ground_truth",
            stratum="miss",
            asset_id="video-a",
            frame_index=3,
            ground_truth_id="ground-3",
        ),
        ReviewUnit(
            unit_id="4" * 64,
            unit_type="unmatched_ground_truth",
            stratum="miss",
            asset_id="video-a",
            frame_index=4,
            ground_truth_id="ground-4",
        ),
    )
    sample = ReviewSample(
        population_counts={
            "clean_match": 80,
            "flagged_or_ambiguous_match": 0,
            "miss": 20,
            "false_positive": 0,
        },
        primary_quota_sample_counts={
            "clean_match": 2,
            "flagged_or_ambiguous_match": 0,
            "miss": 2,
            "false_positive": 0,
        },
        sample_counts={
            "clean_match": 2,
            "flagged_or_ambiguous_match": 0,
            "miss": 2,
            "false_positive": 0,
        },
        units=units,
    )
    verdicts = tuple(
        ReviewVerdict(
            unit_id=unit.unit_id,
            stratum=unit.stratum,
            unit_type=unit.unit_type,
            acceptable_without_redraw=unit.unit_id == "2" * 64,
            severe_drift_or_wrong_target=unit.unit_id == "1" * 64,
        )
        for unit in units
    )

    summary = compute_verdict_summary(sample, verdicts)

    assert summary.raw_challenge_sample is not None
    assert summary.population_weighted is not None
    assert summary.raw_challenge_sample.acceptable_without_redraw == 0.25
    assert summary.population_weighted.acceptable_without_redraw == 0.4
    assert summary.raw_challenge_sample.severe_drift_or_wrong_target == 0.5
    assert summary.population_weighted.severe_drift_or_wrong_target == 0.5
    assert summary.population_weighted.acceptable_denominator == 100
    assert summary.population_weighted.severe_denominator == 80
    assert summary.status == "human_gates_failed"


def test_verdict_contract_rejects_mutually_exclusive_human_labels() -> None:
    with pytest.raises(ValidationError, match="cannot both"):
        ReviewVerdict(
            unit_id="f" * 64,
            stratum="clean_match",
            unit_type="matched_pair",
            acceptable_without_redraw=True,
            severe_drift_or_wrong_target=True,
        )
