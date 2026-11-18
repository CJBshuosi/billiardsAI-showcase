"""Collect three fresh motion clips for offline analysis; no model or production entry point."""

import argparse
import csv
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import table_check as table
from frozen_check import validate_candidate

from billiards_ai.lab.mainline1_usb_capture import enumerate_usb_devices

STAGES = (
    ("slow", "慢球：轻推白球，经过台面中央，尽量让球完整停下"),
    ("fast_cushion", "快球碰库：正常较快击球，让白球碰一次库边；不要跳球或击球出桌"),
    ("pocket", "进袋：选择容易打进的目标球，正常完成一次进袋尝试，没进也保留视频"),
)


def verify_clip(folder):
    """Read all frames and cross-check saved/captured IDs and host timing, not tracking accuracy."""
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "SAMPLE_COLLECTED_REVIEW_REQUIRED":
        raise ValueError("录像未正常完成")
    # Same mode/cadence/control gate as stills, after checking the distinct recording status.
    table.validate_capture({**report, "status": "READBACK_COLLECTED_REVIEW_REQUIRED"})
    archive = report.get("archive", {})
    if archive.get("queue_drops") != 0 or archive.get("writer_error") is not None:
        raise ValueError("录像有丢帧或写入错误，保留证据，不当作完整轨迹样本")
    mode = report.get("observed_after_capture", {})
    if (
        mode.get("frame_width") != 1920
        or mode.get("frame_height") != 1080
        or mode.get("fourcc_text") != "MJPG"
        or not np.isfinite(float(mode.get("fps", "nan")))
        or abs(mode["fps"] - 60) > 0.5
    ):
        raise ValueError("录像结束时采集模式不匹配")
    with (folder / "frames.csv").open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    with (folder / "capture-times.csv").open(encoding="utf-8", newline="") as f:
        captured = list(csv.DictReader(f))
    count = len(saved)
    if not 2 <= count <= 2000 or len(captured) != count or archive.get("saved_frames") != count:
        raise ValueError("录像帧数和时间戳行数不一致")
    times = np.array([float(row["host_perf_counter_seconds"]) for row in saved])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("录像时间戳非有限或不递增")
    for i, (s, c) in enumerate(zip(saved, captured, strict=True)):
        if (
            int(s["saved_index"]) != i
            or int(s["capture_index"]) != i
            or int(c["capture_index"]) != i
            or c["queued_for_video"] != "True"
            or c["host_perf_counter_seconds"] != s["host_perf_counter_seconds"]
        ):
            raise ValueError("录像帧号、采集帧号或时间戳不匹配")
    elapsed = float(times[-1] - times[0])
    fps = (count - 1) / elapsed
    seconds = float(report["record_requested_seconds"])
    if (
        not np.isfinite(seconds)
        or not 0 < seconds <= 30
        or not seconds - 0.1 <= elapsed <= seconds + 1
    ):
        raise ValueError("录像实际时长不足或异常")
    cadence = report.get("cadence_with_recording", {})
    if (
        not 57 <= fps <= 63
        or cadence.get("frames") != count
        or not np.isfinite(float(cadence.get("measured_fps", "nan")))
        or abs(cadence["measured_fps"] - fps) > 0.001
    ):
        raise ValueError("录像实际读帧速度不足或报告与原始时间戳不符")
    video = folder / "camera.avi"
    if not video.is_file() or not 0 < video.stat().st_size <= 512 * 1024 * 1024:
        raise ValueError("录像文件不存在或超过大小限制")
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    try:
        if not capture.isOpened():
            raise ValueError("录像文件不能解码")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape != (1080, 1920, 3):
                raise ValueError("解码帧尺寸错误")
            decoded += 1
            if decoded > count:
                raise ValueError("录像帧数多于索引")
        if decoded != count:
            raise ValueError("录像被截断或帧数少于索引")
    finally:
        capture.release()
    import hashlib

    digest = hashlib.sha256()
    with video.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(
        status="MEDIA_INTEGRITY_VERIFIED_REVIEW_REQUIRED",
        decoded_frames=decoded,
        video_sha256=digest.hexdigest(),
        elapsed_seconds=elapsed,
        measured_recording_fps=fps,
        maximum_read_gap_ms=float(np.diff(times).max() * 1000),
        tracking_evaluated=False,
        production_ready=False,
        acceptance="NOT_EVALUATED",
        timing="host_after_read_not_sensor; AVI nominal timing is not authoritative",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-clip", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)
    if args.audit_clip:
        if not args.audit_output:
            parser.error("--audit-output required")
        try:
            table.dump(args.audit_output, verify_clip(args.audit_clip))
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

    summary = dict(
        status="PARTIAL",
        kind="motion_media_for_offline_analysis",
        clips=[],
        tracking_evaluated=False,
        refit_performed=False,
        production_ready=False,
        active_calibration_written=False,
        absolute_position_validated=False,
        official_results_written=False,
    )
    try:
        manifest = json.loads((root / "packet-manifest.json").read_text(encoding="utf-8"))
        required = {
            "lens-candidate.json",
            "frozen-plane-candidate.json",
            "scripts/diagnostics/raytheon_table/motion_check.py",
            "scripts/diagnostics/raytheon_table/frozen_check.py",
            "scripts/diagnostics/raytheon_table/table_check.py",
            "scripts/diagnostics/raytheon_camera/camera_check.py",
        }
        if not required.issubset(manifest["files"]):
            raise ValueError("诊断包清单不完整")
        for name, info in manifest["files"].items():
            p = root / name
            if (
                not p.resolve().is_relative_to(root)
                or p.is_symlink()
                or camera_check.sha256(p) != info["sha256"]
                or p.stat().st_size != info["bytes"]
            ):
                raise ValueError("诊断包校验失败")
        frozen = json.loads((root / "frozen-plane-candidate.json").read_text(encoding="utf-8"))
        lens = json.loads((root / "lens-candidate.json").read_text(encoding="utf-8"))
        validate_candidate(frozen)
        table.lens_parameters(lens)
        if frozen["lens_file_sha256"] != camera_check.sha256(root / "lens-candidate.json"):
            raise ValueError("冻结平面与镜头不匹配")
        hashes = {}
        for name in ("packet-manifest.json", "lens-candidate.json", "frozen-plane-candidate.json"):
            hashes[name] = camera_check.sha256(root / name)
            with (args.output / name).open("xb") as stream:
                stream.write((root / name).read_bytes())
        summary["frozen_files_sha256"] = hashes
        devices = enumerate_usb_devices(command_runner=partial(subprocess.run, timeout=20))
        table.dump(
            args.output / "devices.json",
            [{**d.model_dump(mode="json"), "fingerprint": d.durable_fingerprint} for d in devices],
        )
        if len(devices) != 1 or devices[0].durable_fingerprint != lens["device_fingerprint"]:
            raise ValueError("须只连接本次RER相机；不能猜测设备对应关系")
        print("关闭 AMCap/相机/OBS。移走棋盘。保持机位、焦点和曝光不变。", flush=True)
        print("本包只录素材，不跑识别、不写正式成绩。共3段，每段20秒。", flush=True)
        print("按Enter后先准备约10秒，看到 RECORDING NOW 或听到提示音才击球。", flush=True)
        for index, (name, instruction) in enumerate(STAGES, 1):
            input(
                f"[{index}/3] {instruction}。"
                "确认相机未动、参数未变、棋盘已移走后按Enter；Ctrl+C取消："
            )
            table.dump(
                args.output / (name + "-operator-confirmation.json"),
                dict(
                    camera_pose_focus_unchanged=True,
                    board_removed=True,
                    instruction=instruction,
                    action_performed="NOT_AUTOMATICALLY_VERIFIED",
                ),
            )
            result = camera_check.run_camera_child(
                args.output / name,
                index=0,
                backend="DSHOW",
                sample_seconds=10,
                record_seconds=20,
                require_mode=True,
                expected_controls=table.EXPECTED_CONTROLS,
            )
            table.dump(args.output / (name + "-execution.json"), result)
            if result["status"] != "COMPLETED":
                raise ValueError("录制失败或超时，停止后续录制，请回传证据")
            print(name + "：录制结束，正在检查文件完整性，请稍等。", flush=True)
            dest = args.output / name / "media-review.json"
            checked = camera_check.run_bounded(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--audit-clip",
                    str(args.output / name),
                    "--audit-output",
                    str(dest),
                ],
                args.output / (name + "-audit.log"),
                timeout=45,
            )
            table.dump(args.output / (name + "-audit-execution.json"), checked)
            if checked["status"] != "COMPLETED":
                raise ValueError("录像完整性检查未通过或超时，保留本轮文件回传")
            media = json.loads(dest.read_text(encoding="utf-8"))
            summary["clips"].append(dict(name=name, **media))
            print(f"[{index}/3] 已保留视频和时间戳。", flush=True)
        for name, digest in hashes.items():
            if camera_check.sha256(root / name) != digest:
                raise ValueError("冻结文件在采集期间发生变化")
        summary["status"] = "COLLECTED_REVIEW_REQUIRED"
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = str(error) or "INTERRUPTED"
        print(summary["error"], flush=True)
    finally:
        print("正在打包视频，文件可能较大，请勿关闭窗口。", flush=True)
        target = camera_check.pack_result(args.output, summary)
        print("SEND BACK: " + str(target), flush=True)
    return 0 if summary["status"] == "COLLECTED_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
