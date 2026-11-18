from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab.events.models import TrackedFrame
from billiards_ai.lab.mainline1_realtime_diagnostics import diagnose_realtime_shot
from billiards_ai.lab.mainline1_realtime_tracker import TrackedObservation
from scripts.algorithm import offline_realtime_diagnostics as public


def raw_diagnostic(size=3):
    return asdict(
        diagnose_realtime_shot(
            shot_id="shot-0001",
            interval_start=0,
            interval_end=size - 1,
            declared_source_count=size,
            shot_start=0,
            shot_end=size - 1,
            source_ids=range(size),
            tracked=tuple(TrackedObservation(i, i, ()) for i in range(size)),
            rule_frames=tuple(TrackedFrame(i, i, ()) for i in range(size)),
            anchors=(),
            publishable=False,
            abstention_reason="tracker_degraded",
            tracker_snapshot=None,
            tracker_config=None,
        )
    )


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def test_actual_runtime_metadata_survives_driver_and_export_boundaries():
    raw = raw_diagnostic()
    envelope = public.project_first_shot_diagnostics(raw)
    assert envelope["status"] == "OBSERVED_METADATA"
    assert envelope["diagnostic"]["publishable"] is False
    assert envelope["diagnostic"]["unknown_group_counts_omitted"] == 0
    # The two boundaries accept Python runtime tuples and serialized JSON lists.
    assert public.validate_first_shot_diagnostics_export(json.loads(encode(envelope))) == envelope
    assert "unknown_group_counts_omitted" not in raw
    assert not any(key in envelope for key in ("complete", "product_pass", "model_pass"))


def test_export_samples_count_rows_and_exposes_omission_without_changing_raw_status():
    raw = raw_diagnostic(public.MAX_FRAMES)
    envelope = public.project_first_shot_diagnostics(raw)
    assert envelope["status"] == "OBSERVED_METADATA"
    diagnostic = envelope["diagnostic"]
    assert len(diagnostic["unknown_group_counts"]) == public.MAX_EXPORTED_COUNT_ROWS
    assert diagnostic["unknown_group_counts_omitted"] == public.MAX_FRAMES - 128
    assert len(encode(envelope)) <= public.MAX_BYTES
    assert public.validate_first_shot_diagnostics_export(envelope) == envelope
    assert len(raw["unknown_group_counts"]) == public.MAX_FRAMES


def test_oversized_enum_is_rejected_before_hashing_or_copying_it():
    class LongReason(str):
        def __hash__(self):
            raise AssertionError("oversized reason must be rejected before hashing")

    raw = raw_diagnostic()
    raw["abstention_reason"] = LongReason("private" * 1000)
    assert public.project_first_shot_diagnostics(raw)["status"] == "INVALID"


def test_missing_and_failed_diagnostics_remain_explicit_and_do_not_export_error_text():
    from billiards_ai.lab.mainline1_realtime_diagnostics import RealtimeShotDiagnostics

    assert public.project_first_shot_diagnostics(None)["status"] == "NOT_CAPTURED"
    raw = asdict(
        RealtimeShotDiagnostics(
            status="UNAVAILABLE",
            shot_id="shot-0001",
            error_type="PrivateClassName",
        )
    )
    result = public.project_first_shot_diagnostics(raw)
    assert result["status"] == "UNAVAILABLE"
    assert result["diagnostic"]["error_type"] == "OTHER"
    assert "PrivateClassName" not in str(result)
    assert public.validate_first_shot_diagnostics_export(result) == result
    for invalid in (False, 0.0):
        result["diagnostic"]["unknown_group_counts_omitted"] = invalid
        assert public.validate_first_shot_diagnostics_export(result)["status"] == "INVALID"


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "reason",
        "shot",
        "count",
        "bool",
        "nan",
        "rows",
        "row_count",
        "tuple_size",
        "hash",
        "index",
        "status",
        "sequence_extra",
        "outside",
        "missing_type",
    ],
)
def test_invalid_or_private_metadata_is_replaced_by_a_small_fixed_envelope(case):
    raw = raw_diagnostic()
    secret = "SECRET_SENTINEL_private_path"
    if case == "extra":
        raw[secret] = secret
    elif case == "reason":
        raw["abstention_reason"] = secret
    elif case == "shot":
        raw["shot_id"] = secret
    elif case == "count":
        raw["source"]["inspected_count"] = 4097
    elif case == "bool":
        raw["interval_start"] = False
    elif case == "nan":
        raw["tracker_policy"] = dict(
            allow_accounted_frame_gaps=True,
            maximum_accounted_frame_gap=8,
            enforce_processing_budget=False,
            processing_budget_ms=float("nan"),
            maximum_source_lag_frames=2,
            history_frames=720,
        )
    elif case == "rows":
        raw["unknown_group_counts"] = [(0, 0, 0)] * 4097
    elif case == "row_count":
        raw["unknown_group_counts"] = [(0, 2, 1)] * 3
    elif case == "tuple_size":
        raw["unknown_group_counts"] = [(0, 0, 0, secret)] * 3
    elif case == "hash":
        raw["tracked"]["inspected_order_sha256"] = secret
    elif case == "index":
        raw["source"]["duplicate_count"] = 1
        raw["source"]["first_duplicate_indices"] = [999]
    elif case == "status":
        raw["status"] = "PRODUCT_PASS"
    elif case == "sequence_extra":
        raw["source"][secret] = secret
    elif case == "outside":
        raw["shot_end"] = 10
    else:
        del raw["source"]["truncated"]
    result = public.project_first_shot_diagnostics(raw)
    assert result == dict(contract=public.CONTRACT, status="INVALID", diagnostic=None)
    assert secret not in str(result)


def test_export_boundary_rejects_added_fields_tampered_status_and_unbounded_preview():
    result = public.project_first_shot_diagnostics(raw_diagnostic(130))
    variants = [copy.deepcopy(result) for _ in range(3)]
    variants[0]["diagnostic"]["private"] = "SECRET_SENTINEL"
    variants[1]["status"] = "UNAVAILABLE"
    variants[2]["diagnostic"]["unknown_group_counts"].append([1, 0, 0])
    for variant in variants:
        assert public.validate_first_shot_diagnostics_export(variant)["status"] == "INVALID"


@pytest.mark.parametrize("room", [0, 1, 70, 200, 20_000])
def test_budget_never_evicts_primary_snapshots_or_failures(room):
    base = {"snapshots": [{"shot": "shot-0001"}], "failures": ["REMOTE_PREFIX"]}
    document = copy.deepcopy(base)
    document["first_shot_diagnostics"] = public.project_first_shot_diagnostics(raw_diagnostic())
    cap = len(encode(base)) + room
    public.fit_first_shot_diagnostics(document, owner=document, render=encode, max_bytes=cap)
    assert document["snapshots"] == base["snapshots"] and document["failures"] == base["failures"]
    assert len(encode(document)) <= cap
    if room == 0:
        assert document == base
    elif room == 20_000:
        assert document["first_shot_diagnostics"]["status"] == "OBSERVED_METADATA"
    elif "first_shot_diagnostics" in document:
        assert document["first_shot_diagnostics"]["status"] == "BUDGET_OMITTED"


def test_optional_budget_guard_does_not_change_an_already_oversized_legacy_result():
    document = {"snapshots": ["existing" * 100], "failures": ["OLD_FAILURE"]}
    original = copy.deepcopy(document)
    public.fit_first_shot_diagnostics(document, owner=document, render=encode, max_bytes=10)
    assert document == original


def test_driver_captures_first_diagnostic_before_eof_without_hiding_failure():
    from billiards_ai.edge.route_assistance.records import FieldSessionStatus
    from scripts.algorithm.run_offline_managed_blackbox import _Actor, _EpisodeFailure

    diagnostic = {"first_shot_diagnostics": None}
    video = {"status": "ready", "reason": None}
    source = SimpleNamespace(
        subsystem_health=lambda: {
            "video_health": video,
            "analysis_health": {"realtime_diagnostics": diagnostic},
        },
        failure_reason=None,
        restart_required=False,
    )
    actor = object.__new__(_Actor)
    actor.source = source
    actor.session = "managed-session"
    actor.state = SimpleNamespace(session_id=actor.session, projection_source=source)
    actor.store = SimpleNamespace(
        load_session=lambda _: SimpleNamespace(status=FieldSessionStatus.OPEN)
    )
    actor.report = {"capture": {}, "fullgame": {}}
    actor.health()
    assert actor.report["capture"]["first_shot_diagnostics"]["status"] == "NOT_CAPTURED"
    diagnostic["first_shot_diagnostics"] = raw_diagnostic()
    video.update(status="degraded", reason="capture source ended unexpectedly")
    with pytest.raises(_EpisodeFailure, match="CAPTURE_EOF"):
        actor.health()
    first = copy.deepcopy(actor.report["capture"]["first_shot_diagnostics"])
    assert first["status"] == "OBSERVED_METADATA"
    diagnostic["first_shot_diagnostics"]["shot_id"] = "shot-0002"
    with pytest.raises(_EpisodeFailure, match="CAPTURE_EOF"):
        actor.health()
    assert actor.report["capture"]["first_shot_diagnostics"] == first
    del actor.report["fullgame"]
    actor.report["capture"].clear()
    with pytest.raises(_EpisodeFailure, match="CAPTURE_EOF"):
        actor.health()
    assert "first_shot_diagnostics" not in actor.report["capture"]


def test_episode_pretty_encoding_budget_omits_only_the_optional_capture_field():
    base = {"capture": {"inference": {}}, "fullgame": {"failures": ["ABSTENTION_EVIDENCE"]}}
    document = copy.deepcopy(base)
    document["capture"]["first_shot_diagnostics"] = public.project_first_shot_diagnostics(
        raw_diagnostic()
    )

    def pretty(value):
        return (json.dumps(value, indent=2) + "\n").encode("utf-8")

    public.fit_first_shot_diagnostics(
        document,
        owner=document["capture"],
        render=pretty,
        max_bytes=len(pretty(base)),
    )
    assert pretty(document) == pretty(base)


def test_independent_diagnostic_survives_episode_budget_omission(tmp_path):
    from scripts.algorithm.run_offline_managed_blackbox import _write_first_shot_diagnostic

    base = {"capture": {}, "fullgame": {"failures": ["CAPTURE_EOF"]}}
    report = copy.deepcopy(base)
    diagnostic = public.project_first_shot_diagnostics(raw_diagnostic())
    report["capture"]["first_shot_diagnostics"] = diagnostic
    assert _write_first_shot_diagnostic(tmp_path, report["capture"])
    public.fit_first_shot_diagnostics(
        report,
        owner=report["capture"],
        render=encode,
        max_bytes=len(encode(base)),
    )
    assert report == base
    path = tmp_path / "first-shot-diagnostics.json"
    assert len(path.read_bytes()) <= public.MAX_BYTES
    assert json.loads(path.read_bytes()) == diagnostic
    # A subsequent write must neither overwrite the retained first file nor change capture state.
    changed = {"first_shot_diagnostics": public.project_first_shot_diagnostics(None)}
    assert not _write_first_shot_diagnostic(tmp_path, changed)
    assert json.loads(path.read_bytes()) == diagnostic
    assert changed["first_shot_diagnostics"]["status"] == "NOT_CAPTURED"


def test_independent_diagnostic_storage_failure_and_legacy_do_not_change_results(tmp_path):
    from scripts.algorithm.run_offline_managed_blackbox import _write_first_shot_diagnostic

    assert not _write_first_shot_diagnostic(tmp_path, {})
    assert not tuple(tmp_path.iterdir())
    capture = {"first_shot_diagnostics": public.project_first_shot_diagnostics(raw_diagnostic())}
    before = copy.deepcopy(capture)
    assert not _write_first_shot_diagnostic(tmp_path / "missing-parent", capture)
    assert capture == before
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("with_diagnostic", [False, True])
def test_episode_budget_and_disk_bytes_share_lf_encoding_on_windows(
    tmp_path,
    monkeypatch,
    with_diagnostic,
):
    from scripts.algorithm.run_offline_managed_blackbox import _write_episode_report

    # UNIT primary data places LF bytes exactly at the unchanged episode cap.
    base = {
        "capture": {},
        "fullgame": {"snapshots": ["UNIT primary row"] * 32, "failures": ["CAPTURE_EOF"]},
        "UNIT_padding": "",
    }

    def pretty(value):
        return (json.dumps(value, indent=2) + "\n").encode("utf-8")

    base["UNIT_padding"] = "x" * (262144 - len(pretty(base)))
    expected = pretty(base)
    assert len(expected) == 262144
    assert len(expected.replace(b"\n", b"\r\n")) > 262144
    report = copy.deepcopy(base)
    if with_diagnostic:
        report["capture"]["first_shot_diagnostics"] = public.project_first_shot_diagnostics(
            raw_diagnostic()
        )
    target = tmp_path / "episode.json"
    original_open = Path.open

    def windows_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if path == target and mode == "w" and newline is None:
            newline = "\r\n"
        return original_open(
            path,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", windows_open)
    _write_episode_report(tmp_path, report)
    assert target.read_bytes() == expected
    assert report == base


def test_episode_primary_over_cap_remains_intact_with_independent_sidecar(tmp_path):
    from scripts.algorithm.run_offline_managed_blackbox import (
        _write_episode_report,
        _write_first_shot_diagnostic,
    )

    base = {"capture": {}, "fullgame": {"failures": ["CAPTURE_EOF"]}, "UNIT_padding": "x" * 262144}
    report = copy.deepcopy(base)
    diagnostic = public.project_first_shot_diagnostics(raw_diagnostic())
    report["capture"]["first_shot_diagnostics"] = diagnostic
    assert _write_first_shot_diagnostic(tmp_path, report["capture"])
    _write_episode_report(tmp_path, report)
    raw = (tmp_path / "episode.json").read_bytes()
    assert len(raw) > 262144
    assert json.loads(raw) == base and report == base
    assert json.loads((tmp_path / "first-shot-diagnostics.json").read_bytes()) == diagnostic
