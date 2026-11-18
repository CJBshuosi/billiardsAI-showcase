"""Two fresh board captures evaluated against a frozen local-metric plane; no fitting."""

import argparse
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import table_check as table

from billiards_ai.lab.mainline1_usb_capture import enumerate_usb_devices


def validate_candidate(candidate):
    if (
        candidate.get("schema_version") != "1.0"
        or candidate.get("status") != "FROZEN_LOCAL_METRIC_CANDIDATE_NOT_ACTIVE"
        or any(
            candidate.get(k) is not False
            for k in (
                "production_ready",
                "active_calibration_written",
                "absolute_position_validated",
            )
        )
        or candidate.get("checkerboard") != dict(columns=11, rows=8, square_size_mm=25)
    ):
        raise ValueError("只能使用冻结的局部距离诊断候选")
    n = table.matrix(candidate.get("normal_camera"), (3,), "冻结平面法向")
    h = float(candidate["camera_distance_mm"])
    if not np.isfinite(h) or h <= 0 or abs(np.linalg.norm(n) - 1) > 1e-8 or n[2] <= 0:
        raise ValueError("冻结平面参数无效")
    placement = table.matrix(candidate.get("homography_for_placement_only"), (3, 3), "区域检查矩阵")
    if abs(np.linalg.det(placement)) < 1e-12:
        raise ValueError("区域检查矩阵退化")
    return n, h, placement


def measure_frozen(image_points, lens, candidate, position):
    if position not in ("center", "corner"):
        raise ValueError("只能拍中央和任一偏角")
    n, h, placement = validate_candidate(candidate)
    points = table.matrix(image_points, (88, 2), "棋盘角点")
    if np.any(points < 0) or np.any(points[:, 0] >= 1920) or np.any(points[:, 1] >= 1080):
        raise ValueError("棋盘角点超出原图")
    k, d = table.lens_parameters(lens)
    und = cv2.undistortPoints(points.reshape(-1, 1, 2), k, d).reshape(-1, 2)
    rays = np.column_stack([und, np.ones(88)])
    denominators = rays @ n
    if not np.isfinite(denominators).all() or np.any(denominators <= 1e-8):
        raise ValueError("棋盘不在有效相机前方平面内")
    xyz = rays * (h / denominators[:, None])
    # Old rail homography is ONLY a coarse region check, never the metric projection.
    corrected = cv2.undistortPoints(points.reshape(-1, 1, 2), k, d, P=k)
    mm = cv2.perspectiveTransform(corrected, placement).reshape(-1, 2)
    if not np.isfinite(mm).all():
        raise ValueError("区域投影无效")
    cx, cy = mm.mean(axis=0) / [2540, 1270]
    inside = bool((mm > 0).all() and (mm[:, 0] < 2540).all() and (mm[:, 1] < 1270).all())
    actual = "other"
    if 0.35 < cx < 0.65 and 0.35 < cy < 0.65:
        actual = "center"
    elif (cx < 0.4 or cx > 0.6) and (cy < 0.4 or cy > 0.6):
        actual = ("lower_" if cy < 0.4 else "upper_") + ("left" if cx < 0.4 else "right")
    matches = inside and (
        actual == "center" if position == "center" else actual not in ("center", "other")
    )
    obj = np.array([(x * 25, y * 25) for y in range(8) for x in range(11)])
    i, j = np.triu_indices(88, 1)
    error = np.linalg.norm(xyz[i] - xyz[j], axis=1) - np.linalg.norm(obj[i] - obj[j], axis=1)
    grid = xyz.reshape(8, 11, 3)
    cells = np.concatenate(
        [
            np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
            np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
        ]
    )
    return dict(
        position=position,
        actual_region=actual,
        placement_matches=bool(matches),
        board_inside_table=inside,
        pair_count=len(error),
        mean_cell_mm=float(cells.mean()),
        max_cell_error_mm=float(abs(cells - 25).max()),
        long_span_mean_mm=float(np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1).mean()),
        short_span_mean_mm=float(np.linalg.norm(grid[-1] - grid[0], axis=1).mean()),
        max_pair_distance_error_mm=float(abs(error).max()),
        rms_pair_distance_error_mm=float(np.sqrt(np.mean(error**2))),
        refit_performed=False,
        absolute_position_validated=False,
        production_ready=False,
        acceptance="NOT_EVALUATED",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "scripts/diagnostics/raytheon_camera"))
    import camera_check

    summary = dict(
        status="PARTIAL",
        kind="fresh_frozen_plane_local_metric_recheck",
        refit_performed=False,
        production_ready=False,
        active_calibration_written=False,
        absolute_position_validated=False,
    )
    try:
        manifest = json.loads((root / "packet-manifest.json").read_text(encoding="utf-8"))
        required = {
            "lens-candidate.json",
            "frozen-plane-candidate.json",
            "scripts/diagnostics/raytheon_table/frozen_check.py",
            "scripts/diagnostics/raytheon_table/board_check.py",
            "scripts/diagnostics/raytheon_table/table_check.py",
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
                or path.stat().st_size != info["bytes"]
            ):
                raise ValueError("诊断包校验失败，请勿混用旧包")
        table.dump(args.output / "packet-manifest.json", manifest)
        lens = json.loads((root / "lens-candidate.json").read_text(encoding="utf-8"))
        candidate = json.loads((root / "frozen-plane-candidate.json").read_text(encoding="utf-8"))
        table.lens_parameters(lens)
        validate_candidate(candidate)
        if candidate.get("lens_file_sha256") != camera_check.sha256(root / "lens-candidate.json"):
            raise ValueError("冻结平面与镜头文件不匹配")
        frozen_hash = camera_check.sha256(root / "frozen-plane-candidate.json")
        summary["frozen_candidate_sha256"] = frozen_hash
        # Preserve exact bytes so the outbound and returned freeze hashes are identical.
        for name in ("lens-candidate.json", "frozen-plane-candidate.json"):
            with (args.output / name).open("xb") as stream:
                stream.write((root / name).read_bytes())
        devices = enumerate_usb_devices(command_runner=partial(subprocess.run, timeout=20))
        table.dump(
            args.output / "devices.json",
            [{**d.model_dump(mode="json"), "fingerprint": d.durable_fingerprint} for d in devices],
        )
        if len(devices) != 1 or devices[0].durable_fingerprint != lens["device_fingerprint"]:
            raise ValueError("须只连接本次RER相机；本包不猜测设备对应关系")
        print(
            "仅复核冻结参数；不改曝光/焦距，不重拟合，不激活标定。关闭 AMCap/相机/OBS。", flush=True
        )
        print("确认相机机位未动、焦距未变。只保留25mm铝板，平放台呢，不手持、不垫高。", flush=True)
        results = []
        for number, (position, label) in enumerate((("center", "中央"), ("corner", "任一偏角")), 1):
            input(
                f"[{number}/2] 把铝板平放在画面{label}，整块入镜，远离袋口；"
                "确认上述条件且手离开后按 Enter（Ctrl+C取消）："
            )
            table.dump(
                args.output / (position + "-operator-confirmation.json"),
                dict(
                    camera_pose_and_focus_unchanged=True,
                    board_flat_on_cloth=True,
                    board_25mm_11x8_only=True,
                    verification="operator_confirmed_not_automatic",
                ),
            )
            folder = args.output / position
            execution = camera_check.run_bounded(
                [
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
                ],
                args.output / (position + "-capture.log"),
                timeout=45,
            )
            table.dump(args.output / (position + "-capture-execution.json"), execution)
            if execution["status"] != "COMPLETED":
                raise ValueError("采集失败或超时；请回传结果包")
            table.validate_capture(json.loads((folder / "report.json").read_text(encoding="utf-8")))
            detected = folder / "board.json"
            execution = camera_check.run_bounded(
                [
                    sys.executable,
                    str(root / "scripts/diagnostics/raytheon_table/board_check.py"),
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
                raise ValueError("自动找板失败或超时；保留原图回传，不用制作十字")
            board = json.loads(detected.read_text(encoding="utf-8"))
            if board.get("source_sha256") != camera_check.sha256(folder / "baseline.png"):
                raise ValueError("检测角点与当前原图不匹配")
            result = measure_frozen(board["image_points"], lens, candidate, position)
            result["source_sha256"] = board["source_sha256"]
            result["frozen_candidate_sha256"] = frozen_hash
            table.dump(folder / "frozen-measurement.json", result)
            results.append(result)
            print(
                f"{label}已采完，88个角点。最大点对距离误差 "
                f"{result['max_pair_distance_error_mm']:.3f} mm（待审阅，不代表通过）。",
                flush=True,
            )
            if not result["placement_matches"]:
                print("摆放区域覆盖不足，已保留证据，请完成剩余步骤后回传。", flush=True)
        if (
            camera_check.sha256(root / "frozen-plane-candidate.json") != frozen_hash
            or camera_check.sha256(root / "lens-candidate.json") != candidate["lens_file_sha256"]
        ):
            raise ValueError("冻结参数文件在采集期间发生变化")
        coverage = (
            "CENTER_AND_CORNER_COLLECTED"
            if all(r["placement_matches"] for r in results)
            else "INCOMPLETE_REVIEW_REQUIRED"
        )
        table.dump(
            args.output / "frozen-review.json",
            dict(
                boards=results,
                coverage=coverage,
                frozen_candidate_sha256=frozen_hash,
                refit_performed=False,
                acceptance="NOT_EVALUATED",
                production_ready=False,
                absolute_position_validated=False,
                active_calibration_written=False,
                limitations=[
                    "local lengths cannot establish absolute position or ball-center plane",
                    "camera stability and board flatness are operator-confirmed only",
                    "pair residuals share corners and are correlated",
                ],
            ),
        )
        summary.update(status="COLLECTED_REVIEW_REQUIRED", coverage=coverage)
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = str(error) or "INTERRUPTED"
        print(summary["error"], flush=True)
    finally:
        target = camera_check.pack_result(args.output, summary)
        print("SEND BACK: " + str(target), flush=True)
    return 0 if summary["status"] == "COLLECTED_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
