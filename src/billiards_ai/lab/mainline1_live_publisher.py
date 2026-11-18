"""Fail-closed publication seam for complete live Mainline 1 shots."""

from __future__ import annotations

import time
from pathlib import Path

from billiards_ai.lab.mainline1_shot_product import Mainline1ShotProductPayloads


def publish_live_shot(
    *,
    product: Mainline1ShotProductPayloads,
    identity: dict[str, object],
    output_root: Path,
):
    """Atomically publish one complete live shot through the canonical publisher."""

    if not product.results:
        raise ValueError("live publication requires one complete shot")
    if len(product.results) != 1 or not product.results[0].window.complete:
        raise ValueError("incomplete live shot cannot be activated")
    if identity.get("source_kind") != "usb_live":
        raise ValueError("live publication requires usb_live source identity")

    from billiards_ai.lab.mainline1_shot_local_video import _publish_shot_local_artifact

    return _publish_shot_local_artifact(
        product=product,
        identity=identity,
        output_root=output_root,
        total_started=time.perf_counter(),
        decode_seconds=0.0,
        detection_seconds=0.0,
        event_seconds=0.0,
        canonical_timing={"stable_track_ms": 0.0, "event_observation_ms": 0.0},
        performance_mode="usb_live_shot",
    )
