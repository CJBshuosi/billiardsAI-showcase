"""Five flat-board placements: local metric diagnostic, no absolute-position acceptance."""

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import time
import webbrowser
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import table_check as table

from billiards_ai.lab.mainline1_lens_sample_capture import (
    CheckerboardSpec,
    detect_checkerboard_sample,
)
from billiards_ai.lab.mainline1_usb_capture import enumerate_usb_devices

SPEC = CheckerboardSpec(columns=11, rows=8, square_size_mm=25.0)
POSITIONS = ("center", "lower_left", "lower_right", "upper_left", "upper_right")


def evaluate_rails(payload, lens):
    if payload.get("confirmed") is not True:
        raise ValueError("请确认相机/机位和尺寸，并确认铝板将平放、不倾斜")
    cal, corners = table.fit_rails(payload.get("rails_px"), lens)
    return {
        "status": "RAILS_ONLY_NOT_VALIDATED",
        "reference_check": "NOT_EVALUATED",
        "homography_undistorted_px_to_mm": cal.matrix,
        "corners_undistorted_px": corners.tolist(),
        "rails_px": payload["rails_px"],
        "operator_confirmed": True,
        "table_length_mm": 2540,
        "table_width_mm": 1270,
        "coordinate_origin": "virtual_lower_left_cushion_nose_intersection",
        "production_ready": False,
        "acceptance": "NOT_EVALUATED",
        "absolute_position_validated": False,
        "active_calibration_written": False,
    }


def detect_board(frame):
    """Propose bright board ROIs automatically; reuse the proven original-pixel SB detector."""
    if frame is None or frame.shape != (1080, 1920, 3):
        raise ValueError("需要1920x1080原始帧")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A white board's disconnected light squares become one bounded search proposal.
    mask = cv2.inRange(gray, 100, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 65 <= w <= 650 and 65 <= h <= 650 and 0.45 < w / h < 2.3 and w * h >= 8000:
            x0, y0 = max(0, x - 24), max(0, y - 24)
            roi = (x0, y0, min(1920, x + w + 24) - x0, min(1080, y + h + 24) - y0)
            candidates.append((w * h, roi))
    for _, roi in sorted(candidates, reverse=True)[:12]:
        sample = detect_checkerboard_sample(frame, SPEC, roi=roi)
        if sample is not None:
            return {
                "image_points": sample.image_points,
                "object_points": sample.object_points,
                "roi": roi,
                "detector": "existing_SB_auto_ROI_original_pixels",
                "checkerboard": {"columns": 11, "rows": 8, "square_size_mm": 25},
                "coordinate_space": "original_full_frame_pixels",
            }
    raise ValueError("未检测到完整11×8内角点棋盘；请保留原图回传，不需要制作十字")


def measure_board(image_points, lens, homography, position):
    if position not in POSITIONS:
        raise ValueError("未知摆板区域")
    points = table.matrix(image_points, (88, 2), "棋盘角点")
    if np.any(points < 0) or np.any(points[:, 0] >= 1920) or np.any(points[:, 1] >= 1080):
        raise ValueError("棋盘角点超出原图")
    k, d = table.lens_parameters(lens)
    h = table.matrix(homography, (3, 3), "台面变换")
    corrected = cv2.undistortPoints(points.reshape(-1, 1, 2), k, d, P=k)
    mm = cv2.perspectiveTransform(corrected, h).reshape(-1, 2)
    if not np.isfinite(mm).all():
        raise ValueError("棋盘投影无效")
    center = mm.mean(axis=0) / [2540, 1270]
    in_table = bool((mm > 0).all() and (mm[:, 0] < 2540).all() and (mm[:, 1] < 1270).all())
    if position == "center":
        matches = bool(np.all(center > 0.35) and np.all(center < 0.65))
    else:
        matches = bool(
            (center[0] < 0.4 if position.endswith("left") else center[0] > 0.6)
            and (center[1] < 0.4 if position.startswith("lower") else center[1] > 0.6)
        )
    object_mm = np.array([(x * 25, y * 25) for y in range(8) for x in range(11)], np.float64)
    i, j = np.triu_indices(88, k=1)
    expected = np.linalg.norm(object_mm[i] - object_mm[j], axis=1)
    observed = np.linalg.norm(mm[i] - mm[j], axis=1)
    error = observed - expected
    grid = mm.reshape(8, 11, 2)
    cell = np.concatenate(
        (
            np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
            np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
        )
    )
    return {
        "position": position,
        "placement_matches": matches and in_table,
        "board_inside_table": in_table,
        "centroid_table_mm": mm.mean(axis=0).tolist(),
        "mapped_corners_mm": mm.tolist(),
        "pair_count": len(error),
        "max_pair_distance_error_mm": float(np.max(np.abs(error))),
        "rms_pair_distance_error_mm": float(np.sqrt(np.mean(error**2))),
        "mean_cell_mm": float(cell.mean()),
        "max_cell_error_mm": float(np.max(np.abs(cell - 25))),
        "long_span_mean_mm": float(np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1).mean()),
        "short_span_mean_mm": float(np.linalg.norm(grid[-1] - grid[0], axis=1).mean()),
        "nominal_cell_mm": 25,
        "nominal_long_span_mm": 250,
        "nominal_short_span_mm": 175,
        "evaluation": "LOCAL_LENGTHS_REVIEW_REQUIRED",
        "absolute_position_validated": False,
        "production_ready": False,
        "acceptance": "NOT_EVALUATED",
    }


def evaluate_all(payload, lens, boards):
    if tuple(board["position"] for board in boards) != POSITIONS:
        raise ValueError("必须包含中央及四个偏角各一次，不能用重复区域替代")
    rails = evaluate_rails(payload, lens)
    results = [
        measure_board(
            b["image_points"], lens, rails["homography_undistorted_px_to_mm"], b["position"]
        )
        for b in boards
    ]
    return {
        **rails,
        "status": "BOARD_LOCAL_LENGTHS_REVIEW_REQUIRED",
        "boards": results,
        "coverage": "FIVE_REGIONS_COLLECTED"
        if all(r["placement_matches"] for r in results)
        else "INCOMPLETE_REVIEW_REQUIRED",
        "board_used_to_fit_homography": False,
        "flatness": "operator_confirmed_not_automatically_verified",
        "limitations": [
            "local lengths cannot validate global translation or absolute position",
            "board thickness, tilt, rail clicks and camera motion affect measurements",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detect-frame", type=Path)
    parser.add_argument("--detection-output", type=Path)
    args = parser.parse_args(argv)
    if args.detect_frame:
        if not args.detection_output:
            parser.error("--detection-output required")
        try:
            data = args.detect_frame.read_bytes()
            result = detect_board(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
            result["source_sha256"] = hashlib.sha256(data).hexdigest()
            table.dump(args.detection_output, result)
            points = np.array(result["image_points"], np.float32).reshape(-1, 1, 2)
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            cv2.drawChessboardCorners(frame, (11, 8), points, True)
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise ValueError("角点证据图编码失败")
            with args.detection_output.with_suffix(".png").open("xb") as stream:
                stream.write(encoded.tobytes())
            return 0
        except Exception as error:
            print(str(error), file=sys.stderr)
            return 2
    if not args.output:
        parser.error("--output required")
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "scripts/diagnostics/raytheon_camera"))
    import camera_check

    summary = {
        "status": "PARTIAL",
        "kind": "flat_board_local_lengths",
        "production_ready": False,
        "active_calibration_written": False,
        "absolute_position_validated": False,
    }
    try:
        manifest = json.loads((root / "packet-manifest.json").read_text(encoding="utf-8"))
        required = {
            "lens-candidate.json",
            "scripts/diagnostics/raytheon_table/board_check.py",
            "scripts/diagnostics/raytheon_table/rails-page.html",
            "scripts/diagnostics/raytheon_camera/camera_check.py",
        }
        if not required.issubset(manifest["files"]):
            raise ValueError("诊断包清单不完整")
        for name, info in manifest["files"].items():
            path = root / name
            if (
                not path.resolve().is_relative_to(root)
                or path.is_symlink()
                or camera_check.sha256(path) != info["sha256"]
            ):
                raise ValueError("诊断包文件校验失败")
        table.dump(args.output / "packet-manifest.json", manifest)
        lens = json.loads((root / "lens-candidate.json").read_text(encoding="utf-8"))
        table.lens_parameters(lens)
        table.dump(args.output / "lens-candidate.json", lens)
        devices = enumerate_usb_devices(command_runner=partial(subprocess.run, timeout=20))
        table.dump(
            args.output / "devices.json",
            [{**d.model_dump(mode="json"), "fingerprint": d.durable_fingerprint} for d in devices],
        )
        if len(devices) != 1 or devices[0].durable_fingerprint != lens["device_fingerprint"]:
            raise ValueError("须只连接本次RER相机，不猜测设备对应关系")
        labels = ("中央", "左下偏角", "右下偏角", "左上偏角", "右上偏角")
        boards = []
        print("只用现有铝板。不需要纸十字、卷尺或输入毫米坐标。关闭 AMCap/相机/OBS。", flush=True)
        for i, (position, label) in enumerate(zip(POSITIONS, labels, strict=True)):
            input(
                f"[{i + 1}/5] 请将铝板平放在画面{label}区域，远离袋口、不要翘起；"
                "手离开后按 Enter（Ctrl+C 取消）："
            )
            folder = args.output / position
            cmd = [
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
                json.dumps(table.EXPECTED_CONTROLS),
                "--output",
                str(folder),
            ]
            execution = camera_check.run_bounded(
                cmd, args.output / (position + "-capture.log"), timeout=45
            )
            table.dump(args.output / (position + "-capture-execution.json"), execution)
            if execution["status"] != "COMPLETED":
                raise ValueError("采集失败或超时；请回传结果包")
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            table.validate_capture(report)
            detected = folder / "board.json"
            execution = camera_check.run_bounded(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--detect-frame",
                    str(folder / "baseline.png"),
                    "--detection-output",
                    str(detected),
                ],
                args.output / (position + "-detect.log"),
                timeout=30,
            )
            table.dump(args.output / (position + "-detect-execution.json"), execution)
            if execution["status"] != "COMPLETED":
                raise ValueError("自动找板失败或超时；请回传原图，不用制作十字")
            board = json.loads(detected.read_text(encoding="utf-8"))
            boards.append({"position": position, **board})
            print(f"{label}：已检测到完整88个内角点。", flush=True)
        table.dump(args.output / "board-detections.json", boards)
        frame = cv2.imdecode(
            np.frombuffer((args.output / "center/baseline.png").read_bytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        state = {}
        token = secrets.token_hex(16)
        server = table.make_server(
            args.output,
            frame,
            lens,
            token,
            state,
            page_name="rails-page.html",
            evaluator=lambda payload, candidate: evaluate_all(payload, candidate, boards),
        )
        server.timeout = 0.5
        url = f"http://127.0.0.1:{server.server_port}/{token}/"
        print("最后只需在全桌图上点8个胶边点。打开：" + url, flush=True)
        try:
            try:
                webbrowser.open(url)
            except webbrowser.Error:
                print("请手动复制上面的地址到本机浏览器。", flush=True)
            deadline = time.monotonic() + 1800
            while not state.get("done"):
                if time.monotonic() > deadline:
                    raise ValueError("页面超时，证据仍保留")
                server.handle_request()
            if state.get("cancelled"):
                raise ValueError("已取消，证据仍保留")
            summary.update(status="COLLECTED_REVIEW_REQUIRED", coverage=state["result"]["coverage"])
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
