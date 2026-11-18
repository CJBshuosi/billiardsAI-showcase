from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

from scripts.diagnostics.package_raytheon_csharp_capture_live import build_packet
from scripts.diagnostics.raytheon_csharp_capture_live.smoke import StreamValidator


def _stream() -> bytes:
    readiness = json.dumps(
        {
            "schema_version": "1.0",
            "kind": "mainline1_native_capture_ready",
            "device_locator": "USB\\VID_32E4&PID_6678&MI_00\\6&B920475&0&0000",
            "native_handle_ref": "dshow://camera-path",
            "durable_fingerprint": "8" * 64,
            "width": 1920,
            "height": 1080,
            "fps_numerator": 60,
            "fps_denominator": 1,
            "pixel_format": "MJPG",
            "controls_locked": True,
        },
        separators=(",", ":"),
    ).encode()
    payload = b"\xff\xd8jpeg-payload\xff\xd9"
    frame = (
        struct.pack(
            ">8sQQIII32s",
            b"BAUVC001",
            0,
            100,
            1920,
            1080,
            len(payload),
            hashlib.sha256(payload).digest(),
        )
        + payload
    )
    return struct.pack(">I", len(readiness)) + readiness + frame


def test_live_validator_checks_identity_mode_and_frame_hash() -> None:
    validator = StreamValidator(
        io.BytesIO(_stream()),
        expected_locator="USB\\VID_32E4&PID_6678&MI_00\\6&B920475&0&0000",
        expected_fingerprint="8" * 64,
    )

    readiness = validator.read_readiness()
    frame_id, timestamp_ns, payload = validator.read_frame()

    assert readiness["controls_locked"] is True
    assert (frame_id, timestamp_ns, payload) == (0, 100, b"\xff\xd8jpeg-payload\xff\xd9")


def test_live_packet_is_scoped_and_non_production(tmp_path: Path) -> None:
    archive = build_packet(Path.cwd(), tmp_path)

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest_name = next(name for name in names if name.endswith("packet-manifest.json"))
        root = manifest_name.removesuffix("packet-manifest.json")
        manifest = json.loads(bundle.read(manifest_name))

        assert manifest["kind"] == "raytheon_csharp_capture_live_packet"
        assert manifest["production_ready"] is False
        assert manifest["mutates_service_or_production_data"] is False
        assert root + "smoke.py" in names
        assert root + "run.ps1" in names
        assert root + "native/windows/mainline1_uvc_capture_csharp/main.cs" in names
        native_source = bundle.read(
            root + "native/windows/mainline1_uvc_capture_csharp/main.cs"
        ).decode("utf-8")
        assert "[In] IBaseFilter source" in native_source
        assert "[In, MarshalAs(UnmanagedType.LPStruct)] Guid interfaceId" in native_source
        assert "FramePacket firstFrame = graph.TakeFrame(5000);" in native_source
        assert "FramePacket frame = graph.TakeFrame(500);" in native_source
        assert archive.name == "raytheon-csharp-capture-live-20260906-v4.zip"
