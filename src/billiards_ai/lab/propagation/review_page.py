"""Static Chinese pages for reviewing one frozen propagation unit at a time."""

from __future__ import annotations

from html import escape
from typing import Literal

from pydantic import Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.evaluation import ReviewStratum, ReviewUnitType


class ReviewPageModel(FrozenContract):
    unit_id: Sha256
    unit_type: ReviewUnitType
    stratum: ReviewStratum
    asset_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    media_url: NonEmptyStr
    evidence_lines: tuple[NonEmptyStr, ...]
    completed_count: int = Field(ge=0)
    total_count: int = Field(gt=0)
    revision: int = Field(ge=0)
    acceptable_without_redraw: bool | None = None
    severe_drift_or_wrong_target: bool | None = None
    note: str = ""
    validation_error: NonEmptyStr | None = None

    @model_validator(mode="after")
    def progress_and_media_are_bound(self) -> ReviewPageModel:
        if self.completed_count >= self.total_count:
            raise ValueError("active review page requires an incomplete sample")
        if self.media_url != f"/media/{self.unit_id}.jpg":
            raise ValueError("review media URL must be derived from the frozen unit identity")
        return self


_UNIT_LABELS: dict[ReviewUnitType, str] = {
    "matched_pair": "匹配预测",
    "unmatched_ground_truth": "漏检",
    "unmatched_prediction": "误检",
}


def _radio_group(*, name: str, title: str, selected: bool | None) -> str:
    true_checked = " checked" if selected is True else ""
    false_checked = " checked" if selected is False else ""
    return (
        f"<fieldset><legend>{title}</legend>"
        f'<label><input required type="radio" name="{name}" value="true"{true_checked}> 是</label>'
        f'<label><input required type="radio" name="{name}" '
        f'value="false"{false_checked}> 否</label>'
        "</fieldset>"
    )


def render_review_page(model: ReviewPageModel) -> str:
    """Render one self-contained review form without scripts or remote assets."""

    evidence = "".join(f"<li>{escape(line)}</li>" for line in model.evidence_lines)
    error = (
        f'<p role="alert" class="error">{escape(model.validation_error)}</p>'
        if model.validation_error
        else ""
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<link rel="stylesheet" href="/assets/review.css">'
        "<title>SAM2 传播证据复核</title></head><body><main>"
        f"<h1>SAM2 传播证据复核</h1><p>第 {model.completed_count + 1} / "
        f"{model.total_count} 项 · {_UNIT_LABELS[model.unit_type]}</p>"
        f"<p>视频：{escape(model.asset_id)} · 帧：{model.frame_index}</p>"
        f'<img src="{escape(model.media_url, quote=True)}" alt="当前冻结复核证据">'
        f"<ul>{evidence}</ul>"
        "<p>规则：两个选项不能同时选择“是”；漏检或误检不能标记为可直接使用。</p>"
        f"{error}"
        '<form method="post" action="/verdict">'
        f'<input type="hidden" name="unit_id" value="{model.unit_id}">'
        f'<input type="hidden" name="revision" value="{model.revision}">'
        + _radio_group(
            name="acceptable_without_redraw",
            title="可直接使用，无需重画",
            selected=model.acceptable_without_redraw,
        )
        + _radio_group(
            name="severe_drift_or_wrong_target",
            title="严重漂移或目标错误",
            selected=model.severe_drift_or_wrong_target,
        )
        + '<label>备注（可选）<textarea name="note" maxlength="1000">'
        + escape(model.note)
        + "</textarea></label>"
        '<button type="submit">保存并查看下一项</button></form></main></body></html>'
    )


def render_complete_page(*, total_count: int, revision: int) -> str:
    """Render the non-passing completion state before immutable export."""

    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<link rel="stylesheet" href="/assets/review.css">'
        "<title>复核完成</title></head><body><main><h1>复核已完成</h1>"
        f"<p>已记录 {total_count} / {total_count} 项，修订号 {revision}。</p>"
        "<p>结果仍需导出并校验不可变 verdict artifact；此页面不声明通过。</p>"
        "</main></body></html>"
    )


REVIEW_CSS = """
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #111827; color: #f9fafb; }
main { max-width: 960px; margin: auto; padding: 1rem; }
img { display: block; max-width: 100%; border: 1px solid #4b5563; }
fieldset, label { display: block; margin: 1rem 0; }
fieldset label { display: inline-block; margin-right: 2rem; }
textarea { display: block; width: 100%; min-height: 5rem; }
button { padding: .7rem 1rem; }
""".strip()


ReviewBooleanName = Literal["acceptable_without_redraw", "severe_drift_or_wrong_target"]
