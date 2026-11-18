"""Load the shared Mainline 1 shot-local policy from YAML."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from billiards_ai.lab.mainline1_shot_local_video import Mainline1ShotLocalConfig
from billiards_ai.lab.shot_local.cue import CueTrackingConfig
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.fragments import FragmentTrackingConfig
from billiards_ai.lab.shot_local.window import ShotWindowConfig


def load_mainline1_shot_local_config(
    path: Path, *, expected_sha256: str | None = None
) -> Mainline1ShotLocalConfig:
    try:
        data = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("shot-local config digest changed")
        payload = yaml.safe_load(data.decode("utf-8"))
        window = payload["window"]
        cue = payload["cue"]
        fragment = payload["fragment"]
        event = payload["event"]
        return Mainline1ShotLocalConfig(
            window=ShotWindowConfig(
                motion_threshold_mm=float(window["motion_threshold_mm"]),
                motion_confirm_frames=int(window["motion_confirm_frames"]),
                still_duration_ms=int(window["still_duration_ms"]),
            ),
            cue=CueTrackingConfig(
                ambiguity_margin=float(cue["ambiguity_margin"]),
                reacquire_confidence_min=float(cue["reacquire_confidence_min"]),
            ),
            fragment=FragmentTrackingConfig(
                motion_threshold_mm=float(fragment["motion_threshold_mm"]),
                continuation_distance_mm=float(fragment["continuation_distance_mm"]),
                ambiguity_mm=float(fragment["ambiguity_mm"]),
            ),
            event=ShotLocalEventConfig(
                first_contact_distance_mm=float(event["first_contact_distance_mm"]),
                first_contact_ambiguity_mm=float(event["first_contact_ambiguity_mm"]),
                cue_motion_min_mm=float(event["cue_motion_min_mm"]),
                contact_time_tolerance_ms=int(event["contact_time_tolerance_ms"]),
                cushion_distance_mm=float(event["cushion_distance_mm"]),
                pocket_distance_mm=float(event["pocket_distance_mm"]),
                pot_missing_frames=int(event["pot_missing_frames"]),
                after_match_distance_mm=float(event["after_match_distance_mm"]),
                table_length_mm=int(event["table_length_mm"]),
                table_width_mm=int(event["table_width_mm"]),
                cushion_normal_motion_min_mm=float(event.get("cushion_normal_motion_min_mm", 3)),
                cushion_max_gap_frames=int(event.get("cushion_max_gap_frames", 3)),
                cushion_sparse_extra_mm=float(event.get("cushion_sparse_extra_mm", 40)),
                cushion_cooldown_frames=int(event.get("cushion_cooldown_frames", 6)),
                collision_distance_mm=float(event.get("collision_distance_mm", 55)),
                collision_motion_min_mm=float(event.get("collision_motion_min_mm", 8)),
                collision_max_gap_frames=int(event.get("collision_max_gap_frames", 3)),
                collision_ambiguity_mm=float(event.get("collision_ambiguity_mm", 5)),
                collision_chain_confidence_min=float(
                    event.get("collision_chain_confidence_min", 0.7)
                ),
            ),
        )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError("shot-local config is invalid") from exc
