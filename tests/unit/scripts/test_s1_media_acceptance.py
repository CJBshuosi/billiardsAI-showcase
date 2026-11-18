"""Offline software/media checks; these do not establish Windows acceptance."""

import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
)
from scripts.development import blackbox_source as sources
from scripts.development.blackbox_report import atomic_json

ONLINE_ID, RAW_ID, EXPORT_ID = "1" * 32, "2" * 32, "3" * 32


def tool():
    path = Path("scripts/development/s1_media_acceptance.py")
    assert path.is_file(), "independent offline media entry is missing"
    return importlib.import_module("scripts.development.s1_media_acceptance")


def owner(path, expected_run_id, **changes):
    atomic_json(
        path,
        dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0, run_id=expected_run_id)
        | changes,
    )


def sign_frozen(frozen):
    frozen.pop("manifest_sha256", None)
    frozen["manifest_sha256"] = hashlib.sha256(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_offline_media_entry_exists_without_starting_product_runtime():
    assert callable(tool().run_stage)


@pytest.fixture
def episode(tmp_path, monkeypatch):
    module = tool()
    source_root = tmp_path / "locked-source"
    source_root.mkdir()
    (source_root / "src").mkdir()
    (source_root / "src/a.py").write_text("VALUE = 1\n")
    (source_root / ".gitattributes").write_bytes(sources.ATTRIBUTES)
    lock = dict(
        contract=sources.CONTRACT,
        commit="a" * 40,
        files={name: sources.pin(source_root / name) for name in ("src/a.py", ".gitattributes")},
    )
    atomic_json(source_root / "SOURCE-LOCK.json", lock)
    monkeypatch.setattr(module, "ROOT", source_root)
    case = tmp_path / "online-owner/case"
    case.mkdir(parents=True)
    recorder = ContinuousSessionRecorder(
        root=case / "raw",
        session_id="session",
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    for index in range(3):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.circle(image, (400 + index * 25, 500), 25, (230, 230, 230), -1)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        recorder.append(
            CaptureFrame("epoch", index, round(index * 1e9 / 60), 1920, 1080, encoded.tobytes())
        )
    recorder.close()
    manifest = recorder.finalized_manifest_path
    raw = json.loads(manifest.read_bytes())
    frozen = dict(
        contract="recording-end/v1",
        session_id="session",
        source_epoch_id="epoch",
        policy="recording-v1",
        end_reason="USER_ENDED_RECORDING",
        cutoff=dict(frame_id=2, captured_at_ns=round(2e9 / 60)),
        raw_manifest_path=str(manifest),
        raw_manifest_sha256=sources.pin(manifest)["sha256"],
        raw_segments=[
            dict(name=s["name"], sha256=s["sha256"])
            | ContinuousSessionRecorder.segment_file_identity(manifest.parent / s["name"])
            for s in raw["segments"]
        ],
        first_frame_id=0,
        last_frame_id=2,
        width=1920,
        height=1080,
        shots=[],
        media_status="RECORDED",
        analysis_status="NOT_REQUESTED",
        encoding_status="NOT_SCHEDULED",
        ended_at_utc="2026-11-18T00:00:00Z",
    )
    sign_frozen(frozen)
    report = dict(
        contract="s1-online-screen-acceptance/v1",
        online_pass=True,
        case_pass=False,
        media_complete=False,
        cleanup_confirmed=True,
        owned_run_id=ONLINE_ID,
        source=dict(kind="locked", producer=lock["commit"], files=lock["files"]),
        frozen=frozen,
        input_manifest=dict(frame_count=3, fps=60, case="inside", intervals=[]),
    )
    atomic_json(case / "local-report.json", report)
    owner(case.parent / "ownership.json", ONLINE_ID)
    return module, case, report


def execute(episode, tmp_path, stage="raw", **kwargs):
    module, case, _ = episode
    output = tmp_path / (stage + "-owner") / "case"
    output.parent.mkdir(exist_ok=True)
    result = module.run_stage(
        case,
        output,
        stage,
        RAW_ID if stage == "raw" else "4" * 32 if stage == "verify" else EXPORT_ID,
        **kwargs,
    )
    return result, output


@pytest.mark.parametrize(
    "changes",
    [
        dict(status="DEADLINE"),
        dict(cleanup_confirmed=False),
        dict(worker_exit_code=1),
        dict(run_id="9" * 32),
    ],
)
def test_failed_or_mismatched_online_owner_cannot_be_rescued(episode, tmp_path, changes):
    _, case, _ = episode
    owner(case.parent / "ownership.json", ONLINE_ID, **changes)
    result, output = execute(episode, tmp_path)
    assert result["stage_pass"] is False
    assert result["failure"]["type"] == "ValueError"
    assert json.loads((output / "local-report.json").read_bytes())["stage_pass"] is False


@pytest.mark.parametrize("change", ["online_failure", "source", "manifest", "cutoff", "path"])
def test_rejects_unqualified_or_tampered_online_inputs(episode, tmp_path, change):
    _, case, report = episode
    if change == "online_failure":
        report["online_pass"] = False
    elif change == "source":
        report["source"]["producer"] = "b" * 40
    elif change == "manifest":
        Path(report["frozen"]["raw_manifest_path"]).write_text("{}")
    elif change == "cutoff":
        report["frozen"]["cutoff"]["frame_id"] = 1
        sign_frozen(report["frozen"])
    else:
        report["frozen"]["raw_manifest_path"] = str(tmp_path / "outside.json")
        sign_frozen(report["frozen"])
    atomic_json(case / "local-report.json", report)
    result, _ = execute(episode, tmp_path)
    assert result["stage_pass"] is False


def test_raw_stage_reads_real_pixels_and_preserves_original_files(episode, tmp_path):
    _, case, report = episode
    before = {str(p): sources.pin(p) for p in case.rglob("*") if p.is_file()}
    result, output = execute(episode, tmp_path)
    assert result["stage_pass"] is True
    assert result["raw_recording"]["decoded_frames"] == 3
    assert result["raw_recording"]["width"] == 1920
    assert result["source"] == report["source"]
    assert result["online_report_pin"] == before[str(case / "local-report.json")]
    assert result["raw_manifest_pin"] == before[report["frozen"]["raw_manifest_path"]]
    assert before == {str(p): sources.pin(p) for p in case.rglob("*") if p.is_file()}
    assert not (output / "independent-original.mp4").exists()


def test_export_requires_raw_success_and_external_owner_success(episode, tmp_path):
    result, _ = execute(episode, tmp_path, "export")
    assert result["stage_pass"] is False
    raw, raw_output = execute(episode, tmp_path)
    assert raw["stage_pass"] is True
    owner(raw_output.parent / "ownership.json", RAW_ID, status="DEADLINE")
    other = tmp_path / "another-export"
    result = episode[0].run_stage(
        episode[1], other, "export", EXPORT_ID, raw_report=raw_output / "local-report.json"
    )
    assert result["stage_pass"] is False
    assert not (other / "independent-original.mp4").exists()


def test_export_rejects_online_report_changes_after_raw_audit(episode, tmp_path):
    result, output = execute(episode, tmp_path)
    assert result["stage_pass"] is True
    owner(output.parent / "ownership.json", RAW_ID)
    _, case, report = episode
    report["input_manifest"]["case"] = "outside"
    atomic_json(case / "local-report.json", report)
    result, exported = execute(episode, tmp_path, "export", raw_report=output / "local-report.json")
    assert result["stage_pass"] is False
    assert not (exported / "independent-original.mp4").exists()


def test_export_uses_actual_frozen_media_and_existing_avc_codec(episode, tmp_path):
    raw, output = execute(episode, tmp_path)
    assert raw["stage_pass"] is True
    owner(output.parent / "ownership.json", RAW_ID)
    raw_report = output / "local-report.json"
    result, exported = execute(episode, tmp_path, "export", raw_report=raw_report)
    assert result["stage_pass"] is True, result.get("failure")
    assert result["raw_report_pin"] == sources.pin(raw_report)
    video_pin = sources.pin(exported / "independent-original.mp4")
    assert result["independent_original"] == dict(
        byte_size=video_pin["bytes"], content_sha256=video_pin["sha256"]
    )
    capture = cv2.VideoCapture(str(exported / "independent-original.mp4"))
    images = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            images.append(image)
    finally:
        capture.release()
    assert len(images) == 3 and images[0].shape == (1080, 1920, 3)
    assert not np.array_equal(images[0], images[-1])


def test_raw_internal_deadline_error_is_recorded_and_original_reader_restored(episode, tmp_path):
    from scripts.algorithm import run_recording_first_fullgame as existing

    original = existing.FrozenCaptureInterval
    failure = TimeoutError("controlled cooperative deadline")

    def check():
        raise failure

    result, _ = execute(episode, tmp_path, continue_check=check)
    assert result["stage_pass"] is False
    assert result["failure"] == dict(type="TimeoutError", message=str(failure))
    assert existing.FrozenCaptureInterval is original


def test_timeout_during_real_raw_iteration_preserves_exception_and_restores_reader(episode):
    from scripts.algorithm import run_recording_first_fullgame as existing

    module, case, report = episode
    original = existing.FrozenCaptureInterval
    failure = TimeoutError("raw frame deadline")
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise failure

    with pytest.raises(TimeoutError) as caught:
        module._verify_raw(report["frozen"], case, 3, check)
    assert caught.value is failure
    assert existing.FrozenCaptureInterval is original


def test_raw_rejects_modified_segment_without_rewriting_original(episode, tmp_path):
    _, case, report = episode
    path = Path(report["frozen"]["raw_manifest_path"])
    segment = json.loads(path.read_bytes())["segments"][0]["name"]
    target = path.parent / segment
    target.write_bytes(b"corrupt raw pixels")
    result, _ = execute(episode, tmp_path)
    assert result["stage_pass"] is False
    assert target.read_bytes() == b"corrupt raw pixels"
    assert (case / "local-report.json").is_file()


def test_source_change_during_real_raw_iteration_cannot_be_claimed_as_pass(episode, tmp_path):
    module, _, _ = episode
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if checks == 5:
            (module.ROOT / "src/a.py").write_text("CHANGED = 2\n")

    result, _ = execute(episode, tmp_path, continue_check=check)
    assert result["stage_pass"] is False
    assert result["raw_recording"]["decoded_frames"] == 3


def test_real_encoder_release_error_keeps_identity_and_diagnostic_error_receipts(
    episode, tmp_path, monkeypatch
):
    from scripts.development.s1_probe_diagnostics import ProbeDiagnostics

    module, _, report = episode
    original = cv2.VideoWriter
    failure = RuntimeError("controlled real writer release failure")

    class Writer:
        def __init__(self, *args):
            self.original = original(*args)

        def __getattr__(self, name):
            return getattr(self.original, name)

        def release(self):
            self.original.release()
            raise failure

    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    output = tmp_path / "observed-export"
    output.mkdir()
    with ProbeDiagnostics(output, EXPORT_ID) as diagnostics:
        with pytest.raises(RuntimeError) as caught:
            module.export_original(report["frozen"], output, 3, diagnostics=diagnostics)
    assert caught.value is failure
    events = json.loads((output / "local-stages.json").read_bytes())["events"]
    assert [(row["stage"], row["event"]) for row in events][-2:] == [
        ("original_writer_release", "error"),
        ("original_encode", "error"),
    ]
    capture = cv2.VideoCapture(str(output / "independent-original.mp4"))
    try:
        assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()


def test_original_export_fsync_uses_a_real_writable_descriptor(episode, tmp_path, monkeypatch):
    module, _, report = episode
    original = os.fsync

    def windows_write_handle_requirement(descriptor):
        # A zero-byte write is inert, and still rejects a read-only descriptor.
        os.write(descriptor, b"")
        return original(descriptor)

    monkeypatch.setattr(module.os, "fsync", windows_write_handle_requirement)
    output = tmp_path / "writable-export"
    output.mkdir()
    result = module.export_original(report["frozen"], output, 3)
    assert result["byte_size"] == (output / "independent-original.mp4").stat().st_size


def encoded_case(episode, tmp_path, **kwargs):
    raw, raw_output = execute(episode, tmp_path)
    assert raw["stage_pass"] is True, raw.get("failure")
    owner(raw_output.parent / "ownership.json", RAW_ID)
    raw_path = raw_output / "local-report.json"
    encoded, output = execute(episode, tmp_path, "encode", raw_report=raw_path, **kwargs)
    assert encoded["stage_pass"] is True, encoded.get("failure")
    owner(output.parent / "ownership.json", EXPORT_ID)
    return encoded, output, raw_path


def test_real_encode_and_full_verify_are_separate_linked_stages(episode, tmp_path):
    from billiards_ai.lab.mainline1_recording_media import _verify_mp4

    calls = []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _verify_mp4.__code__:
            calls.append("full_decode")

    sys.setprofile(observe)
    try:
        encoded, output, raw_path = encoded_case(episode, tmp_path)
        assert calls == [], "encoding cannot consume its budget on full output verification"
        assert encoded["verification_complete"] is False
        video = output / "independent-original.mp4"
        assert encoded["encoded_original"] == dict(
            **sources.pin(video), fps=pytest.approx(60.0, abs=0.00001), frame_count=3
        )
        result, verified = execute(
            episode,
            tmp_path,
            "verify",
            raw_report=raw_path,
            encoded_report=output / "local-report.json",
        )
    finally:
        sys.setprofile(previous)
    assert calls == ["full_decode"]
    assert result["stage_pass"] is True, result.get("failure")
    assert result["verification_complete"] is True
    assert result["encoded_report_pin"] == sources.pin(output / "local-report.json")
    assert result["independent_original"] == dict(
        byte_size=sources.pin(video)["bytes"], content_sha256=sources.pin(video)["sha256"]
    )
    assert not (verified / "independent-original.mp4").exists(), "verify must not encode/copy again"


@pytest.mark.parametrize(
    "fault", ["owner", "media", "raw_link", "stage", "source", "path", "symlink"]
)
def test_verify_refuses_failed_or_tampered_encoded_predecessor(episode, tmp_path, fault):
    encoded, output, raw_path = encoded_case(episode, tmp_path)
    report_path = output / "local-report.json"
    video = output / "independent-original.mp4"
    if fault == "owner":
        owner(output.parent / "ownership.json", EXPORT_ID, status="FAIL")
    elif fault == "media":
        video.write_bytes(b"modified output")
    elif fault == "raw_link":
        encoded["raw_report_pin"]["sha256"] = "f" * 64
    elif fault == "stage":
        encoded["stage_pass"] = False
    elif fault == "source":
        encoded["source"]["producer"] = "f" * 40
    elif fault == "path":
        encoded["encoded_original"]["path"] = str(tmp_path / "unowned.mp4")
    else:
        other = tmp_path / "unowned.mp4"
        video.rename(other)
        video.symlink_to(other)
    if fault in {"raw_link", "stage", "source", "path"}:
        atomic_json(report_path, encoded)
    result, _ = execute(
        episode,
        tmp_path,
        "verify",
        raw_report=raw_path,
        encoded_report=report_path,
    )
    assert result["stage_pass"] is False
    assert result.get("verification_complete") is not True


def source_revision(episode, tmp_path, monkeypatch):
    module, case, report = episode
    capture_root = module.ROOT
    for name in ("s1_media_acceptance.py", "s1_screen_control.py"):
        target = capture_root / "scripts/development" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VERSION = 'capture'\n")
    lock = json.loads((capture_root / "SOURCE-LOCK.json").read_bytes())
    lock["files"] = {
        p.relative_to(capture_root).as_posix(): sources.pin(p)
        for p in capture_root.rglob("*")
        if p.is_file() and p.name != "SOURCE-LOCK.json"
    }
    atomic_json(capture_root / "SOURCE-LOCK.json", lock)
    report["source"] = dict(kind="locked", producer=lock["commit"], files=lock["files"])
    atomic_json(case / "local-report.json", report)
    raw, raw_output = execute(episode, tmp_path)
    assert raw["stage_pass"] is True, raw.get("failure")
    owner(raw_output.parent / "ownership.json", RAW_ID)
    execution_root = tmp_path / "execution-source"
    shutil.copytree(capture_root, execution_root)
    for name in ("s1_media_acceptance.py", "s1_screen_control.py"):
        (execution_root / "scripts/development" / name).write_text("VERSION = 'offline fix'\n")
    lock["commit"] = "b" * 40
    lock["files"] = {name: sources.pin(execution_root / name) for name in lock["files"]}
    atomic_json(execution_root / "SOURCE-LOCK.json", lock)
    monkeypatch.setattr(module, "ROOT", execution_root)
    return capture_root, execution_root, raw_output / "local-report.json"


def test_offline_only_revision_reuses_capture_and_raw_without_relabeling(
    episode, tmp_path, monkeypatch
):
    module, _, online = episode
    capture_root, execution_root, raw_path = source_revision(episode, tmp_path, monkeypatch)
    applicability = module.source_applicability(capture_root)
    assert applicability["capture_source"] == online["source"]
    assert applicability["source"]["producer"] == "b" * 40
    assert applicability["changed_files"] == [
        "scripts/development/s1_media_acceptance.py",
        "scripts/development/s1_screen_control.py",
    ]
    assert applicability["capture_source_lock_pin"] == sources.pin(
        capture_root / "SOURCE-LOCK.json"
    )
    assert applicability["execution_source_lock_pin"] == sources.pin(
        execution_root / "SOURCE-LOCK.json"
    )
    encoded, output = execute(
        episode,
        tmp_path,
        "encode",
        raw_report=raw_path,
        capture_source_root=capture_root,
    )
    assert encoded["stage_pass"] is True, encoded.get("failure")
    owner(output.parent / "ownership.json", EXPORT_ID)
    result, _ = execute(
        episode,
        tmp_path,
        "verify",
        raw_report=raw_path,
        capture_source_root=capture_root,
        encoded_report=output / "local-report.json",
    )
    assert result["stage_pass"] is True, result.get("failure")
    assert result["capture_source"] == online["source"]
    assert result["source"] == applicability["source"]
    assert json.loads(raw_path.read_bytes())["source"] == online["source"]


@pytest.mark.parametrize("fault", ["runtime", "added", "removed"])
def test_source_applicability_rejects_any_other_file_or_inventory_change(
    episode, tmp_path, monkeypatch, fault
):
    module, _, _ = episode
    capture_root, execution_root, _ = source_revision(episode, tmp_path, monkeypatch)
    if fault == "runtime":
        (execution_root / "src/a.py").write_text("UNRELATED = 2\n")
    elif fault == "added":
        (execution_root / "src/added.py").write_text("VALUE = 2\n")
    else:
        (execution_root / "src/a.py").unlink()
    lock = json.loads((execution_root / "SOURCE-LOCK.json").read_bytes())
    lock["files"] = {
        p.relative_to(execution_root).as_posix(): sources.pin(p)
        for p in execution_root.rglob("*")
        if p.is_file() and p.name != "SOURCE-LOCK.json"
    }
    atomic_json(execution_root / "SOURCE-LOCK.json", lock)
    with pytest.raises(ValueError, match="source|inventory|offline"):
        module.source_applicability(capture_root)


def test_encode_alone_can_use_170_seconds_while_verify_keeps_160(episode, tmp_path, monkeypatch):
    from billiards_ai.lab import mainline1_recording_media as media

    module, _, _ = episode
    raw, raw_output = execute(episode, tmp_path)
    assert raw["stage_pass"] is True
    owner(raw_output.parent / "ownership.json", RAW_ID)
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    original_encode, original_verify = media._encode_interval, media._verify_mp4
    verification_errors = []

    def encode(*args, **kwargs):
        now[0] = 161.0
        return original_encode(*args, **kwargs)

    def verify(*args, **kwargs):
        now[0] = 161.0
        try:
            return original_verify(*args, **kwargs)
        except Exception as exc:
            verification_errors.append(exc)
            raise

    monkeypatch.setattr(media, "_encode_interval", encode)
    encoded, output = execute(
        episode, tmp_path, "encode", raw_report=raw_output / "local-report.json"
    )
    assert encoded["stage_pass"] is True, encoded.get("failure")
    owner(output.parent / "ownership.json", EXPORT_ID)
    now[0] = 0.0
    monkeypatch.setattr(media, "_verify_mp4", verify)
    result, _ = execute(
        episode,
        tmp_path,
        "verify",
        raw_report=raw_output / "local-report.json",
        encoded_report=output / "local-report.json",
    )
    assert result["stage_pass"] is False
    # The existing hash helper wraps OSError (including TimeoutError). Preserve
    # that original error contract rather than changing product media code.
    assert isinstance(verification_errors[0].__cause__, TimeoutError)
    assert result["failure"]["type"] == "RecordingPlaybackError"
