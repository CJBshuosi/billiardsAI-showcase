"""An explicitly selected external codec reaches only the owned worker."""

import hashlib
import json
from types import SimpleNamespace

import cv2
import pytest

from scripts.algorithm import recording_windows_entry as entry
from scripts.algorithm import run_recording_first_fullgame as runner


def _selected(tmp_path, monkeypatch, *, enabled=True):
    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    dll, license_text = b"reviewed external codec fixture", b"complete codec license fixture"
    meta = dict(
        filename="openh264-2.5.0-win64.dll",
        bytes=len(dll),
        sha256=hashlib.sha256(dll).hexdigest(),
        license_bytes=len(license_text),
        license_sha256=hashlib.sha256(license_text).hexdigest(),
    )
    monkeypatch.setattr(entry, "CODEC", meta, raising=False)
    codec = tmp_path / "codec"
    codec.mkdir()
    (codec / meta["filename"]).write_bytes(dll)
    (codec / "BINARY_LICENSE.txt").write_bytes(license_text)
    (tmp_path / "PREPARED.json").write_text(json.dumps(dict(manifest_sha256="a" * 64)))
    (tmp_path / "codec-enabled.json").write_text(
        json.dumps(
            dict(
                contract="recording-codec-selection/v1",
                enabled=enabled,
                packet_manifest_sha256="a" * 64,
            )
        )
    )
    return codec / meta["filename"]


def test_selected_codec_is_pinned_and_scoped_without_inheriting_host_secrets(tmp_path, monkeypatch):
    path = _selected(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENH264_LIBRARY", "untrusted-host-library")
    monkeypatch.setenv("OPENAI_API_KEY", "private-host-secret")
    env = entry.child_environment(tmp_path)
    assert env.get("OPENH264_LIBRARY") == str(path)
    assert "private-host-secret" not in json.dumps(env)
    assert "untrusted-host-library" not in json.dumps(env)


@pytest.mark.parametrize("mutation", ["dll", "license", "missing", "symlink", "manifest", "flag"])
def test_invalid_external_codec_never_reaches_owned_environment(tmp_path, monkeypatch, mutation):
    path = _selected(tmp_path, monkeypatch)
    if mutation == "dll":
        path.write_bytes(b"x" * path.stat().st_size)
    elif mutation == "license":
        path.with_name("BINARY_LICENSE.txt").write_bytes(b"changed notice")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "symlink":
        target = path.with_name("other.dll")
        path.rename(target)
        path.symlink_to(target)
    else:
        selection = json.loads((tmp_path / "codec-enabled.json").read_bytes())
        selection["packet_manifest_sha256" if mutation == "manifest" else "enabled"] = (
            "b" * 64 if mutation == "manifest" else "true"
        )
        (tmp_path / "codec-enabled.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        entry.child_environment(tmp_path)


def test_operator_can_disable_external_codec_without_loading_or_retaining_host_choice(
    tmp_path, monkeypatch
):
    path = _selected(tmp_path, monkeypatch, enabled=False)
    path.unlink()
    monkeypatch.setenv("OPENH264_LIBRARY", "untrusted-host-library")
    assert "OPENH264_LIBRARY" not in entry.child_environment(tmp_path)


def test_codec_startup_failure_reports_before_entering_product(tmp_path, monkeypatch):
    path = _selected(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENH264_LIBRARY", str(path))
    entered = []
    monkeypatch.setattr(runner, "main", lambda args: entered.append(args) or 0)
    # Windows codec unavailability is below the real preflight, at the OpenCV OS boundary.
    monkeypatch.setattr(
        cv2, "VideoWriter", lambda *a: SimpleNamespace(isOpened=lambda: False, release=lambda: None)
    )
    assert entry.worker(tmp_path, "a" * 64) == 1
    assert entered == []
    result = json.loads((tmp_path / "codec/preflight.json").read_bytes())
    assert result == dict(
        contract="recording-codec-preflight/v1",
        status="FAIL",
        failure_code="H264_ENCODER_UNAVAILABLE",
        decoded_frames=0,
    )
    assert not (tmp_path / "recording").exists()


def test_real_h264_preflight_removes_only_owned_probe_then_enters_product(tmp_path, monkeypatch):
    path = _selected(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENH264_LIBRARY", str(path))
    sentinel = tmp_path / "codec/keep.txt"
    sentinel.write_bytes(b"keep me")
    entered = []
    monkeypatch.setattr(runner, "main", lambda args: entered.append(args) or 17)
    assert entry.worker(tmp_path, "a" * 64) == 17
    assert len(entered) == 1
    result = json.loads((tmp_path / "codec/preflight.json").read_bytes())
    assert result["status"] == "PASS" and result["decoded_frames"] == 2
    assert sentinel.read_bytes() == b"keep me"
    assert not list((tmp_path / "codec").glob("*.mp4"))


def test_parent_sanitizes_codec_failure_instead_of_waiting_for_product_report(
    tmp_path, monkeypatch
):
    import sys

    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    (tmp_path / "codec").mkdir()

    def spawn(*args, **kwargs):
        (tmp_path / "codec/preflight.json").write_text(
            json.dumps(
                dict(
                    contract="recording-codec-preflight/v1",
                    status="FAIL",
                    failure_code="H264_ENCODER_UNAVAILABLE",
                    decoded_frames=0,
                    private_path="must not return",
                    private_message="must not return",
                )
            )
        )
        return object()

    def owned(api, spawn, persist, metadata, release, **kwargs):
        spawn()
        return dict(status="FAIL", cleanup_confirmed=True, worker_exit_code=1)

    monkeypatch.setattr(entry.subprocess, "Popen", spawn)
    monkeypatch.setitem(
        sys.modules, "ownership", SimpleNamespace(WindowsAPI=object, run_owned=owned)
    )
    manifest = dict(source_commit="a" * 40, input_sha256="b" * 64)
    assert entry.launch(tmp_path, "f" * 64, manifest) == 1
    summary = json.loads((tmp_path / "output/summary.json").read_bytes())
    assert summary["failure_type"] == "CODEC_PREFLIGHT_FAILED"
    assert summary["codec_preflight"] == dict(
        status="FAIL", failure_code="H264_ENCODER_UNAVAILABLE", decoded_frames=0
    )
    assert "must not return" not in json.dumps(summary)
    assert not summary["product_pass"]


def test_non_h264_backend_output_cannot_pass_startup_check(tmp_path, monkeypatch):
    path = _selected(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENH264_LIBRARY", str(path))
    original = cv2.VideoWriter
    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        lambda path, fourcc, fps, geometry: original(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, geometry
        ),
    )
    entered = []
    monkeypatch.setattr(runner, "main", lambda args: entered.append(args) or 0)
    assert entry.worker(tmp_path, "a" * 64) == 1
    assert entered == []
    result = json.loads((tmp_path / "codec/preflight.json").read_bytes())
    assert result["failure_code"] == "H264_READBACK_FAILED"


def test_incomplete_codec_receipt_cannot_break_or_pass_parent_summary(tmp_path):
    (tmp_path / "codec").mkdir()
    (tmp_path / "codec/preflight.json").write_text(
        json.dumps(dict(contract="recording-codec-preflight/v1", status="PASS", decoded_frames=2))
    )
    result = dict(product_pass=True)
    entry._include_codec_summary(tmp_path, result)
    assert not result["product_pass"]
    assert result["failure_type"] == "CODEC_PREFLIGHT_RESULT_INVALID"


def test_two_frame_probe_rejects_output_flush_failure_before_product(tmp_path, monkeypatch):
    import billiards_ai.lab.mainline1_recording_media as media

    path = _selected(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENH264_LIBRARY", str(path))
    entered = []
    monkeypatch.setattr(runner, "main", lambda args: entered.append(args) or 0)

    def denied_flush(path):
        raise OSError("private failure detail must not leave the worker")

    monkeypatch.setattr(media, "_fsync_file", denied_flush)
    assert entry.worker(tmp_path, "a" * 64) == 1
    assert entered == []
    result = json.loads((tmp_path / "codec/preflight.json").read_bytes())
    assert result["failure_code"] == "OUTPUT_FLUSH_FAILED"
    summary = dict(product_pass=False)
    entry._include_codec_summary(tmp_path, summary)
    assert summary["codec_preflight"]["failure_code"] == "OUTPUT_FLUSH_FAILED"
    assert "private failure" not in json.dumps(summary)
    assert not list((tmp_path / "codec").glob("*.mp4"))
