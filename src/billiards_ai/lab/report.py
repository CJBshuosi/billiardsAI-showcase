"""Generate a compact local HTML report with auditable evidence links."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any


def _display_value(value: Any) -> str:
    if value is None:
        return "unsupported"
    if isinstance(value, bool):
        return str(value).lower()
    return escape(str(value))


def build_report_html(predictions: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> str:
    metric_rows = "".join(
        f"<li>{escape(name)}: {_display_value(value)}</li>"
        for name, value in sorted(metrics.items())
    )
    event_rows: list[str] = []
    evidence_ids: set[int] = set()
    for shot in predictions:
        for event in shot.get("events", []):
            ids = event.get("evidence_frame_ids", [])
            if not ids:
                raise ValueError("displayed events require evidence links")
            evidence_ids.update(ids)
            links = " ".join(
                f'<a href="#evidence-frame-{frame_id}">frame {frame_id}</a>' for frame_id in ids
            )
            status = "supported" if event.get("supported") else "unsupported"
            reason = event.get("abstention_reason")
            reason_text = f" ({escape(str(reason))})" if reason else ""
            event_rows.append(
                "<li>"
                f"{escape(str(event['type']))}: {status}{reason_text}; "
                f"confidence={_display_value(event.get('confidence'))}; evidence={links}"
                "</li>"
            )
    evidence_rows = "".join(
        f'<li id="evidence-frame-{frame_id}">evidence frame {frame_id}</li>'
        for frame_id in sorted(evidence_ids)
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Algorithm run</title>'
        "</head><body><h1>Algorithm run</h1><h2>Metrics</h2><ul>"
        f"{metric_rows}</ul><h2>Events</h2><ul>{''.join(event_rows)}</ul>"
        f"<h2>Evidence</h2><ul>{evidence_rows}</ul></body></html>"
    )
