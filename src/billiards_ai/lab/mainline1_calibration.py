"""Local four-corner calibration for the temporary-eye Mainline 1 runner."""

# The self-contained browser page intentionally keeps its HTML/JS readable as a literal.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import shutil
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np

from billiards_ai.lab.mainline1_capture import CaptureSetup, capture_setup_fingerprint
from billiards_ai.shared.calibration.homography import (
    TableCalibration,
    calibrate,
    validate_reference_accuracy,
)
from billiards_ai.shared.models.calibration import (
    CalibrationReference,
    DistortionParameters,
)


class CalibrationInputError(ValueError):
    """The selected frame or its clicked corner set is unsafe."""


@dataclass(frozen=True)
class CalibrationReferenceFrame:
    frame: np.ndarray
    frame_index: int


@dataclass(frozen=True)
class Mainline1CalibrationArtifact:
    path: Path
    artifact_id: str
    calibration: TableCalibration

    @property
    def calibration_path(self) -> Path:
        return self.path / "calibration.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration_reference_frame(video: Path) -> CalibrationReferenceFrame:
    source = video.resolve()
    if video.is_symlink() or not source.is_file():
        raise CalibrationInputError("calibration video is missing or unsafe")
    capture = cv2.VideoCapture(str(source))
    try:
        frame_count = max(0, round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        candidates = tuple(
            dict.fromkeys(
                value
                for value in (
                    frame_count // 2,
                    frame_count // 4,
                    (frame_count * 3) // 4,
                    0,
                )
                if 0 <= value < max(frame_count, 1)
            )
        )
        for frame_index in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                return CalibrationReferenceFrame(frame=frame, frame_index=frame_index)
    finally:
        capture.release()
    raise CalibrationInputError("no calibration frame could be decoded")


def _normalise_corners(
    corners_px: Sequence[Sequence[float]], *, width: int, height: int
) -> tuple[tuple[float, float], ...]:
    if len(corners_px) != 4:
        raise CalibrationInputError("exactly four calibration corners are required")
    try:
        corners = tuple((float(item[0]), float(item[1])) for item in corners_px)
    except (IndexError, TypeError, ValueError) as exc:
        raise CalibrationInputError("calibration corners must be numeric x/y pairs") from exc
    if len(set(corners)) != 4:
        raise CalibrationInputError("calibration corners must be unique")
    if any(
        not np.isfinite((x, y)).all() or x < 0 or y < 0 or x >= width or y >= height
        for x, y in corners
    ):
        raise CalibrationInputError("calibration corner is outside the source frame")
    area_twice = abs(
        sum(
            corners[index][0] * corners[(index + 1) % 4][1]
            - corners[(index + 1) % 4][0] * corners[index][1]
            for index in range(4)
        )
    )
    if area_twice < max(2.0, width * height * 0.02):
        raise CalibrationInputError("calibration corners are degenerate or out of order")
    return corners


def create_calibration_artifact(
    *,
    video: Path,
    frame_index: int,
    frame: np.ndarray,
    corners_px: Sequence[Sequence[float]],
    device_id: str,
    output_root: Path,
    capture_setup: CaptureSetup,
    distortion: DistortionParameters | None = None,
    references: tuple[CalibrationReference, ...] = (),
) -> Mainline1CalibrationArtifact:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise CalibrationInputError("calibration frame must be a non-empty BGR image")
    if frame_index < 0 or not device_id.strip():
        raise CalibrationInputError("frame index and device id are required")
    height, width = frame.shape[:2]
    if (capture_setup.geometry.width, capture_setup.geometry.height) != (width, height):
        raise CalibrationInputError("capture setup geometry does not match the reference frame")
    if capture_setup.camera_device_id != device_id.strip():
        raise CalibrationInputError("capture setup camera does not match the calibration device")
    source_video = video.resolve()
    if video.is_symlink() or not source_video.is_file():
        raise CalibrationInputError("calibration video is missing or unsafe")
    corners = _normalise_corners(corners_px, width=width, height=height)
    calibration = calibrate(device_id.strip(), list(corners), distortion=distortion)
    maximum_reference_error_mm = (
        validate_reference_accuracy(calibration, references) if references else None
    )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise CalibrationInputError("reference frame could not be encoded")
    identity = {
        "schema_version": "1.0",
        "kind": "mainline1_four_corner_calibration",
        "device_id": device_id.strip(),
        "source_video": video.name,
        "frame_index": frame_index,
        "frame_width": width,
        "frame_height": height,
        "corners_px": [list(item) for item in corners],
        "reference_frame_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
        "capture_setup_id": capture_setup.capture_setup_id,
        "capture_setup_fingerprint": capture_setup_fingerprint(capture_setup),
        "frame_geometry": capture_setup.geometry.model_dump(mode="json"),
        "source_video_sha256": _sha256(source_video),
    }
    if distortion is not None or references:
        identity.update(
            {
                "distortion": calibration.distortion.model_dump(mode="json"),
                "reference_count": len(references),
                "references": [item.model_dump(mode="json") for item in references],
                "maximum_reference_error_mm": maximum_reference_error_mm,
            }
        )
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / artifact_id
    if destination.exists():
        expected = {"calibration.json", "manifest.json", "reference-frame.jpg"}
        if {item.name for item in destination.iterdir()} != expected:
            raise CalibrationInputError("existing calibration artifact is incomplete")
        stored = TableCalibration.model_validate_json(
            (destination / "calibration.json").read_bytes()
        )
        return Mainline1CalibrationArtifact(destination, artifact_id, stored)

    temporary = root / f".pending-{uuid4().hex}"
    try:
        temporary.mkdir()
        (temporary / "reference-frame.jpg").write_bytes(encoded.tobytes())
        (temporary / "calibration.json").write_bytes(
            _canonical_bytes(calibration.model_dump(mode="json"))
        )
        inventory = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _sha256(temporary / name),
            }
            for name in ("calibration.json", "reference-frame.jpg")
        }
        (temporary / "manifest.json").write_bytes(
            _canonical_bytes({**identity, "artifact_id": artifact_id, "inventory": inventory})
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return Mainline1CalibrationArtifact(destination, artifact_id, calibration)


def _calibration_page(*, width: int, height: int) -> bytes:
    payload = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>主线一台面四角标定</title><style>
body{{margin:0;background:#0f172a;color:#e5e7eb;font:16px system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:24px}}
h1{{margin:0 0 8px}}.hint{{color:#fbbf24;margin:8px 0 18px}}canvas{{width:100%;height:auto;border:1px solid #475569;background:#000;cursor:crosshair}}
.bar{{display:flex;gap:12px;align-items:center;margin-top:16px}}button{{padding:12px 20px;border:0;border-radius:8px;font-weight:700}}
#save{{background:#22c55e}}#reset{{background:#334155;color:white}}#status{{white-space:pre-wrap}}ol{{line-height:1.7}}
</style></head><body><main><h1>主线一台面四角标定</h1>
<p>请点击四个角袋的<strong>袋口中心</strong>，必须严格按下列顺序：</p>
<ol><li>左下</li><li>右下</li><li>右上</li><li>左上</li></ol>
<p class="hint">四个点应沿台面外围组成不交叉的四边形；点错可重置。</p>
<canvas id="canvas" width="{width}" height="{height}"></canvas>
<div class="bar"><button id="reset">重置</button><button id="save" disabled>确认并保存标定</button><span id="status">等待点击左下角袋中心</span></div>
<script>
const labels=['左下','右下','右上','左上']; const canvas=document.querySelector('#canvas'); const ctx=canvas.getContext('2d');
const image=new Image(); const points=[]; const status=document.querySelector('#status'); const save=document.querySelector('#save');
function draw(){{ctx.drawImage(image,0,0,canvas.width,canvas.height); if(points.length){{ctx.strokeStyle='#22d3ee';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)ctx.lineTo(points[i].x,points[i].y);if(points.length===4)ctx.closePath();ctx.stroke();}} points.forEach((p,i)=>{{ctx.fillStyle='#ef4444';ctx.beginPath();ctx.arc(p.x,p.y,10,0,Math.PI*2);ctx.fill();ctx.fillStyle='white';ctx.font='bold 22px sans-serif';ctx.fillText(`${{i+1}} ${{labels[i]}}`,p.x+14,p.y-14);}});}}
image.onload=draw; image.src='/frame.jpg';
canvas.onclick=(event)=>{{if(points.length===4)return;const rect=canvas.getBoundingClientRect();points.push({{x:(event.clientX-rect.left)*canvas.width/rect.width,y:(event.clientY-rect.top)*canvas.height/rect.height}});draw();save.disabled=points.length!==4;status.textContent=points.length===4?'请检查四点与连线，确认后保存':`等待点击${{labels[points.length]}}角袋中心`;}};
document.querySelector('#reset').onclick=()=>{{points.length=0;save.disabled=true;status.textContent='等待点击左下角袋中心';draw();}};
save.onclick=async()=>{{save.disabled=true;status.textContent='正在保存…';try{{const response=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{corners_px:points.map(p=>[p.x,p.y])}})}});const result=await response.json();if(!response.ok)throw new Error(result.error||'保存失败');status.textContent=`保存成功：\n${{result.calibration}}\n可以关闭此页面。`;}}catch(error){{status.textContent=error.message;save.disabled=false;}}}};
</script></main></body></html>"""
    return payload.encode("utf-8")


def serve_calibration(
    *,
    video: Path,
    frame: np.ndarray,
    frame_index: int,
    device_id: str,
    output_root: Path,
    port: int,
    capture_setup: CaptureSetup,
    browser_opener: Callable[[str], Any] = webbrowser.open,
) -> Mainline1CalibrationArtifact:
    if not 1 <= port <= 65535:
        raise CalibrationInputError("calibration port is invalid")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise CalibrationInputError("reference frame could not be encoded")
    page = _calibration_page(width=frame.shape[1], height=frame.shape[0])
    result: dict[str, Mainline1CalibrationArtifact] = {}

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/":
                self._reply(200, "text/html; charset=utf-8", page)
            elif path == "/frame.jpg":
                self._reply(200, "image/jpeg", encoded.tobytes())
            else:
                self._reply(404, "text/plain; charset=utf-8", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/save":
                self._reply(404, "application/json", b'{"error":"not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 65536:
                    raise CalibrationInputError("invalid request length")
                payload = json.loads(self.rfile.read(length))
                artifact = create_calibration_artifact(
                    video=video,
                    frame_index=frame_index,
                    frame=frame,
                    corners_px=payload.get("corners_px", ()),
                    device_id=device_id,
                    output_root=output_root,
                    capture_setup=capture_setup,
                )
                result["artifact"] = artifact
                body = _canonical_bytes(
                    {"status": "PASS", "calibration": str(artifact.calibration_path)}
                )
                self._reply(200, "application/json; charset=utf-8", body)
            except (CalibrationInputError, json.JSONDecodeError, TypeError) as exc:
                self._reply(
                    400,
                    "application/json; charset=utf-8",
                    _canonical_bytes({"error": str(exc), "status": "FAIL"}),
                )

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 0.5
    url = f"http://127.0.0.1:{port}/"
    print(f"Calibration page: {url}", flush=True)
    browser_opener(url)
    try:
        while "artifact" not in result:
            server.handle_request()
    finally:
        server.server_close()
    return result["artifact"]
