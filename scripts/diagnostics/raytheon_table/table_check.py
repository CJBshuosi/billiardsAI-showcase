"""Non-authoritative RER table geometry check; never writes a calibration registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from billiards_ai.shared.calibration.homography import calibrate, reference_errors_mm
from billiards_ai.shared.models.calibration import CalibrationReference
from billiards_ai.shared.models.geometry import Point

EXPECTED_CONTROLS = {
    "EXPOSURE": -8,
    "AUTOFOCUS": 2,
    "AUTO_WB": 0,
    "FOCUS": 100,
    "WB_TEMPERATURE": 4650,
    "ZOOM": 100,
    "GAIN": 140,
}
EXPECTED_CONTROLS = {key.lower(): value for key, value in EXPECTED_CONTROLS.items()}
LENGTH, WIDTH = 2540, 1270


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def matrix(value, shape, label):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{label}: 数量、形状或数值不合法")
    return a


def lens_parameters(lens):
    if (lens.get("width"), lens.get("height")) != (1920, 1080):
        raise ValueError("镜头参数分辨率不匹配")
    if (
        lens.get("status") != "DIAGNOSTIC_CANDIDATE_NOT_REGISTERED"
        or lens.get("production_ready") is not False
    ):
        raise ValueError("只接受本次未注册的镜头候选")
    k = matrix(lens.get("camera_matrix"), (3, 3), "camera matrix")
    d = matrix(lens.get("coefficients"), (5,), "coefficients")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("镜头内参无效")
    return k, d


def fit_rails(rails_px, lens):
    """Fit only rail points, never independent reference or checkerboard points."""
    k, d = lens_parameters(lens)
    rails = matrix(rails_px, (8, 2), "胶边点击")
    if np.any(rails < 0) or np.any(rails[:, 0] >= 1920) or np.any(rails[:, 1] >= 1080):
        raise ValueError("点击超出原图")
    corrected = cv2.undistortPoints(rails.reshape(-1, 1, 2), k, d, P=k).reshape(-1, 2)
    lines = []
    for a, b in corrected.reshape(4, 2, 2):
        if np.linalg.norm(a - b) < 100:
            raise ValueError("每条胶边两次点击距离太近，请分布在该边两端的直线段")
        line = np.cross([*a, 1], [*b, 1])
        line /= np.linalg.norm(line[:2])
        lines.append(line)
    corners = []
    for a, b in ((3, 0), (0, 1), (1, 2), (2, 3)):
        h = np.cross(lines[a], lines[b])
        if abs(h[2]) < 1e-6:
            raise ValueError("相邻胶边平行或顺序错误")
        corners.append(h[:2] / h[2])
    corners = np.asarray(corners)
    if (
        np.any(corners < 0)
        or np.any(corners[:, 0] >= 1920)
        or np.any(corners[:, 1] >= 1080)
        or not cv2.isContourConvex(corners.astype(np.float32))
        or abs(cv2.contourArea(corners.astype(np.float32))) < 1920 * 1080 * 0.2
        or not (
            corners[0, 0] < corners[1, 0]
            and corners[3, 0] < corners[2, 0]
            and corners[0, 1] > corners[3, 1]
            and corners[1, 1] > corners[2, 1]
        )
    ):
        raise ValueError("四条胶边不构成合法全桌，请按下、右、上、左顺序重选")
    # Coordinates are ALREADY undistorted here. NONE describes only this intermediate
    # plane; no verified OPENCV flag is fabricated and no TableCalibration is exported.
    cal = calibrate("rer-diagnostic-undistorted-plane", [tuple(p) for p in corners], LENGTH, WIDTH)
    return cal, corners


def evaluate(payload, lens):
    """Correct raw points first; reuse the existing homography and error evaluator."""
    if payload.get("confirmed") is not True:
        raise ValueError("请先确认本次相机/尺寸和五个点的独立实测坐标")
    k, d = lens_parameters(lens)
    cal, corners = fit_rails(payload.get("rails_px"), lens)
    refs = matrix(payload.get("references_px"), (5, 2), "参考点点击")
    mm = matrix(payload.get("references_mm"), (5, 2), "参考点毫米坐标")
    if np.any(refs < 0) or np.any(refs[:, 0] >= 1920) or np.any(refs[:, 1] >= 1080):
        raise ValueError("点击超出原图")
    if (
        np.any(mm <= 0)
        or np.any(mm[:, 0] >= LENGTH)
        or np.any(mm[:, 1] >= WIDTH)
        or not np.array_equal(mm, np.round(mm))
    ):
        raise ValueError("参考坐标须为台面内部的整数毫米")
    for points in (refs, mm):
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2) + np.eye(5) * 1e9
        if distances.min() < 20:
            raise ValueError("五个参考点须相互分开，不能重复")
    if cv2.contourArea(cv2.convexHull(mm.astype(np.float32))) < LENGTH * WIDTH * 0.25:
        raise ValueError("参考点覆盖不足，请包含中央和四个偏角区域")
    corrected_refs = cv2.undistortPoints(refs.reshape(-1, 1, 2), k, d, P=k).reshape(-1, 2)
    references = tuple(
        CalibrationReference(pixel=tuple(p), table=Point(x_mm=int(q[0]), y_mm=int(q[1])))
        for p, q in zip(corrected_refs, mm, strict=True)
    )
    errors = reference_errors_mm(cal, references)
    predicted = cv2.perspectiveTransform(corrected_refs.reshape(-1, 1, 2), cal.as_array()).reshape(
        -1, 2
    )
    return {
        "schema_version": "1.0",
        "status": "DIAGNOSTIC_TABLE_REVIEW_ONLY",
        "reference_check": "WITHIN_20MM_REVIEW_REQUIRED" if max(errors) <= 20 else "FAILED",
        "max_reference_error_mm": max(errors),
        "reference_errors_mm": list(errors),
        "predicted_reference_mm": predicted.tolist(),
        "table_length_mm": LENGTH,
        "table_width_mm": WIDTH,
        "coordinate_origin": "virtual_lower_left_cushion_nose_intersection",
        "input_pixel_space": "raw_1920x1080",
        "corners_undistorted_px": corners.tolist(),
        "homography_undistorted_px_to_mm": cal.matrix,
        "clicks_and_measurements": payload,
        "acceptance": "NOT_EVALUATED",
        "production_ready": False,
        "active_calibration_written": False,
    }


def validate_capture(report):
    if (
        report.get("status") != "READBACK_COLLECTED_REVIEW_REQUIRED"
        or report.get("mode_matches_request") is not True
        or report.get("decoded_geometry") != [1920, 1080]
        or report.get("sample_completed") is not True
        or report.get("cadence_without_recording", {}).get("within_5_percent") is not True
    ):
        raise ValueError("采集模式或实测帧率不符合既有60fps候选配置")
    for key in ("controls_after_mode", "controls_after_capture"):
        if any(report.get(key, {}).get(name) != value for name, value in EXPECTED_CONTROLS.items()):
            raise ValueError("相机参数与镜头采样时不一致；不自动调参")


def make_server(
    output, frame, lens, token, state, port=0, *, page_name="page.html", evaluator=evaluate
):
    if frame.shape != (1080, 1920, 3):
        raise ValueError("原图必须为1920x1080")
    prefix = "/" + token + "/"
    page = Path(__file__).with_name(page_name).read_bytes()
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("原始帧编码失败")
    with (output / "reference-frame.png").open("xb") as stream:
        stream.write(encoded.tobytes())

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def reply(self, status, kind, data):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == prefix:
                self.reply(200, "text/html; charset=utf-8", page)
            elif path == prefix + "frame.png":
                self.reply(200, "image/png", encoded.tobytes())
            else:
                self.reply(404, "text/plain", b"not found")

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in (prefix + "save", prefix + "cancel"):
                self.reply(404, "application/json", b"{}")
                return
            try:
                if state.get("done"):
                    raise ValueError("本次已经结束，请勿重复保存")
                origin = self.headers.get("Origin")
                if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
                    raise ValueError("非法请求来源")
                if self.headers.get("Content-Type") != "application/json":
                    raise ValueError("需要JSON请求")
                n = int(self.headers.get("Content-Length", "0"))
                if not 1 <= n <= 32768:
                    raise ValueError("请求大小无效")
                payload = json.loads(self.rfile.read(n))
                if not isinstance(payload, dict):
                    raise ValueError("需要对象")
                if path.endswith("/cancel"):
                    dump(output / "cancelled.json", {"status": "CANCELLED"})
                    state.update(done=True, cancelled=True)
                    self.reply(200, "application/json", b"{}")
                    return
                result = evaluator(payload, lens)
                result["source_frame_sha256"] = hashlib.sha256(encoded.tobytes()).hexdigest()
                result["lens_candidate_sha256"] = hashlib.sha256(
                    json.dumps(lens, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                dump(output / "table-review.json", result)
                state.update(done=True, result=result)
                self.reply(200, "application/json", json.dumps(result, ensure_ascii=False).encode())
            except (ValueError, TypeError, KeyError, OSError, cv2.error) as error:
                self.reply(
                    400,
                    "application/json",
                    json.dumps({"error": str(error)}, ensure_ascii=False).encode(),
                )

        def log_message(self, *args):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "scripts/diagnostics/raytheon_camera"))
    import camera_check

    from billiards_ai.lab.mainline1_usb_capture import enumerate_usb_devices

    summary = {"status": "PARTIAL", "production_ready": False, "active_calibration_written": False}
    try:
        manifest = json.loads((root / "packet-manifest.json").read_text())
        for name, info in manifest["files"].items():
            path = root / name
            if (
                not path.resolve().is_relative_to(root)
                or path.is_symlink()
                or camera_check.sha256(path) != info["sha256"]
            ):
                raise ValueError("诊断包文件校验失败")
        dump(args.output / "packet-manifest.json", manifest)
        lens = json.loads((root / "lens-candidate.json").read_text())
        lens_parameters(lens)
        dump(args.output / "lens-candidate.json", lens)
        devices = enumerate_usb_devices()
        dump(
            args.output / "devices.json",
            [{**d.model_dump(mode="json"), "fingerprint": d.durable_fingerprint} for d in devices],
        )
        if len(devices) != 1 or devices[0].durable_fingerprint != lens["device_fingerprint"]:
            raise ValueError("须只连接镜头采样时的RER摄像头；不猜测设备身份")
        summary["identity_binding"] = "enumeration_plus_operator_confirmation_not_trusted_handle"
        print(
            "Prepare FIVE measured flat reference marks. "
            "Close AMCap/Camera/OBS. Keep camera fixed.",
            flush=True,
        )
        input("Press ENTER when the table and all five marks are ready: ")
        worker = args.output / "capture"
        command = [
            sys.executable,
            str(root / "scripts/diagnostics/raytheon_camera/camera_check.py"),
            "--worker",
            "--index",
            "0",
            "--backend",
            "DSHOW",
            "--width",
            "1920",
            "--height",
            "1080",
            "--require-mode",
            "--record-seconds",
            "0",
            "--sample-seconds",
            "5",
            "--expected-controls",
            json.dumps(EXPECTED_CONTROLS),
            "--output",
            str(worker),
        ]
        execution = camera_check.run_bounded(command, args.output / "capture.log", timeout=45)
        dump(args.output / "worker-execution.json", execution)
        if execution["status"] != "COMPLETED":
            raise ValueError("采集子进程失败或超时，已保留日志")
        report = json.loads((worker / "report.json").read_text())
        validate_capture(report)
        frame = cv2.imdecode(
            np.frombuffer((worker / "baseline.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        state = {}
        token = secrets.token_hex(16)
        server = make_server(args.output, frame, lens, token, state)
        url = f"http://127.0.0.1:{server.server_port}/{token}/"
        server.timeout = 0.5
        print("Open this local page: " + url, flush=True)
        try:
            webbrowser.open(url)
            deadline = time.monotonic() + 1800
            while not state.get("done"):
                if time.monotonic() > deadline:
                    raise ValueError("页面30分钟未完成，本次取消")
                server.handle_request()
            if state.get("cancelled"):
                raise ValueError("用户取消，原图仍会打包")
            summary["status"] = "COLLECTED_REVIEW_REQUIRED"
            summary["reference_check"] = state["result"]["reference_check"]
        finally:
            server.server_close()
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = str(error) or "INTERRUPTED"
        print(summary["error"], flush=True)
    finally:
        target = camera_check.pack_result(args.output, summary)
        print("SEND BACK: " + str(target), flush=True)
    return 0 if summary["status"] == "COLLECTED_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
