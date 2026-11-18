"""Compare two verified RF-DETR checkpoints on the three frozen RER videos.

This is an offline detector diagnostic. It never opens a camera, loads an
identity head, imports the field service, or writes an official result store.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

PACKET_VERSION = "raytheon-rfdetr-compare-20260905-v1"
BASELINE = "9a144955377b289ce47de71769d1305dd8585de4"
RESULT_DIRECTORY_NAME = "rer-table-result-20260905-170622-db9f91f8"
MEDIA_ARCHIVE_NAME = "SEND_BACK-rer-table-result-20260905-170622-db9f91f8.zip"
MEDIA_ARCHIVE_SHA256 = "54d6a6287709d1e3a722c35797ee119f8b029e439498580617620c860cf1b1e1"
MEDIA_SHA256 = {
    "slow": "5c1bc571dcf3265fed659428950a497b055a71a7f31ee6c649304973cfba8b56",
    "fast_cushion": "6317635d4a84023f61ceb3f3cb567aa388e674f6c383a467c8df89047a3be930",
    "pocket": "63666f022d46602f374521db7f718c70d7e5402e929f1144bab4394a8e6ac160",
}
MODEL_SPECS = (
    {
        "label": "bootstrap-2759358",
        "run_id": "967612dd02c72a6af9149b6790820c4e998068df4536d3b006b917c774ac4a51",
        "relative_path": (
            "bootstrap-training-runs/"
            "2759358fe07f948a3c056a18f2b592e0b0ebad76364159f0a774c4eca70ca186/"
            "rfdetr-seg-small-bootstrap-short-r1/"
            "967612dd02c72a6af9149b6790820c4e998068df4536d3b006b917c774ac4a51"
        ),
        "manifest_sha256": "355acf72c8e698e03a1f90265537a2e78e4cd883a1df93e8be8e40376e60934b",
        "checkpoint_sha256": "d9cd5efe6c6deb8b7762e08c0f720cb264a2f41bce9174ddc77154a56f434541",
    },
    {
        "label": "bootstrap-2f96d6b",
        "run_id": "49aac2dc919e2228ef7b937ff426fd31ce2df09365e4002b6bc867da98266d58",
        "relative_path": (
            "bootstrap-training-runs/"
            "2f96d6b0b80fc53f0b3b5d980fa9319413d1214d4db6873ff57d070ebe58ab03/"
            "rfdetr-seg-small-bootstrap-short-r1/"
            "49aac2dc919e2228ef7b937ff426fd31ce2df09365e4002b6bc867da98266d58"
        ),
        "manifest_sha256": "41f886042389e9bc5190dda65f367480690e61d144145fdd8f1b1155bd477aac",
        "checkpoint_sha256": "9739a36001bcb405f18213f5141afc1049437f09a8490956a9f78f43252dd20f",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            return True
        if getattr(metadata, "st_file_attributes", 0) & 0x400:
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _path_chain_is_safe(path: Path) -> bool:
    try:
        absolute = Path(os.path.abspath(path))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            if _is_link(current):
                return False
            if not current.exists():
                break
        return True
    except (OSError, ValueError):
        return False


def choose_frame_indices(
    frame_count: int,
    *,
    peak_index: int,
    stride: int = 10,
    dense_radius: int = 60,
) -> tuple[int, ...]:
    if frame_count <= 0 or stride <= 0 or dense_radius < 0:
        raise ValueError("invalid frame-selection input")
    peak = min(max(int(peak_index), 0), frame_count - 1)
    selected = set(range(0, frame_count, stride))
    selected.update(range(max(0, peak - dense_radius), min(frame_count, peak + dense_radius + 1)))
    selected.update((0, frame_count - 1))
    return tuple(sorted(selected))


def verify_model_run(root: Path, spec: dict) -> dict:
    root = Path(root)
    if not _path_chain_is_safe(root) or _is_link(root):
        raise ValueError("model run has an unsafe path")
    if not root.is_dir() or root.name != spec["run_id"]:
        raise ValueError("model run directory is missing or changed")
    manifest_path = root / "training-manifest.json"
    checkpoint = root / "checkpoint_best.pth"
    for path in (manifest_path, checkpoint):
        if not _path_chain_is_safe(path) or _is_link(path) or not path.is_file():
            raise ValueError("model run contains an unsafe or missing file")
    manifest_hash = sha256(manifest_path)
    checkpoint_hash = sha256(checkpoint)
    if manifest_hash != spec["manifest_sha256"]:
        raise ValueError("training manifest hash mismatch")
    if checkpoint_hash != spec["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("training manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("run_hash") != spec["run_id"]
        or manifest.get("model_class") != "RFDETRSegSmall"
        or not isinstance(manifest.get("artifacts"), dict)
        or manifest["artifacts"].get("checkpoint_best.pth") != checkpoint_hash
    ):
        raise ValueError("training manifest and checkpoint relation changed")
    return {
        "run_id": spec["run_id"],
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


@dataclass
class MediaResolution:
    paths: dict[str, Path]
    source_kind: str
    temporary: tempfile.TemporaryDirectory | None = None

    @property
    def temporary_root(self) -> Path | None:
        return Path(self.temporary.name) if self.temporary is not None else None

    def close(self) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None


def _find_exact(downloads: Path, name: str, *, directory: bool) -> list[Path]:
    matches = []
    entries = 0
    for current_raw, directory_names, file_names in os.walk(downloads, followlinks=False):
        current = Path(current_raw)
        kept = []
        for child_name in sorted(directory_names):
            child = current / child_name
            if not _is_link(child):
                kept.append(child_name)
            if directory and child_name == name and _path_chain_is_safe(child):
                matches.append(child)
        directory_names[:] = kept
        if not directory:
            for file_name in file_names:
                if file_name == name:
                    path = current / file_name
                    if _path_chain_is_safe(path) and not _is_link(path):
                        matches.append(path)
        entries += len(directory_names) + len(file_names)
        if entries > 200_000:
            raise ValueError("Downloads search entry limit exceeded")
    return matches


def _verified_paths(root: Path, media_sha256: dict[str, str]) -> dict[str, Path]:
    paths = {name: root / name / "camera.avi" for name in media_sha256}
    for name, path in paths.items():
        if not _path_chain_is_safe(path) or _is_link(path) or not path.is_file():
            raise ValueError(f"{name} video is missing or unsafe")
        if sha256(path) != media_sha256[name]:
            raise ValueError(f"{name} video hash mismatch")
    return paths


def resolve_media(
    downloads: Path,
    *,
    result_directory_name: str = RESULT_DIRECTORY_NAME,
    archive_name: str = MEDIA_ARCHIVE_NAME,
    archive_sha256: str = MEDIA_ARCHIVE_SHA256,
    media_sha256: dict[str, str] = MEDIA_SHA256,
) -> MediaResolution:
    downloads = Path(downloads)
    if not _path_chain_is_safe(downloads) or not downloads.is_dir():
        raise ValueError("Downloads root is missing or unsafe")
    directories = _find_exact(downloads, result_directory_name, directory=True)
    if directories:
        return MediaResolution(
            paths=_verified_paths(directories[0], media_sha256),
            source_kind="existing_result_directory",
        )
    archives = _find_exact(downloads, archive_name, directory=False)
    if not archives:
        raise ValueError("the frozen motion result directory or SEND_BACK archive was not found")
    archive = archives[0]
    if sha256(archive) != archive_sha256:
        raise ValueError("motion SEND_BACK archive hash mismatch")
    temporary = tempfile.TemporaryDirectory(prefix="rfdetr-media-", dir=downloads)
    temporary_root = Path(temporary.name)
    expected_members = {name: f"{result_directory_name}/{name}/camera.avi" for name in media_sha256}
    try:
        with zipfile.ZipFile(archive) as source:
            if source.testzip() is not None:
                raise ValueError("motion SEND_BACK archive CRC failed")
            names = set(source.namelist())
            for clip, member in expected_members.items():
                if member not in names:
                    raise ValueError(f"{clip} video is absent from the motion archive")
                info = source.getinfo(member)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("motion archive contains a linked video")
                target = temporary_root / clip / "camera.avi"
                target.parent.mkdir(parents=True)
                with source.open(info) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        paths = _verified_paths(temporary_root, media_sha256)
        return MediaResolution(
            paths=paths,
            source_kind="verified_send_back_archive",
            temporary=temporary,
        )
    except Exception:
        temporary.cleanup()
        raise


def predict_frame(model, image, *, threshold: float) -> list[dict]:
    import cv2

    predictions = model.predict(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        threshold=threshold,
        include_source_image=False,
    )
    result = []
    for index in range(len(predictions)):
        box = [float(value) for value in predictions.xyxy[index]]
        confidence = float(predictions.confidence[index])
        class_id = int(predictions.class_id[index])
        if (
            len(box) != 4
            or not all(math.isfinite(value) for value in (*box, confidence))
            or class_id != 0
            or confidence < 0
            or confidence > 1
        ):
            raise ValueError("RF-DETR prediction violates the ball detection boundary")
        result.append({"bbox_xyxy": box, "class_id": class_id, "confidence": confidence})
    return sorted(result, key=lambda item: (*item["bbox_xyxy"], item["confidence"]))


def load_cuda_checkpoint(checkpoint: Path, factory):
    return factory(str(checkpoint), device="cuda:0")


def cuda_model_evidence(model) -> dict[str, str]:
    context = getattr(model, "model", None)
    declared = str(getattr(context, "device", ""))
    module = getattr(context, "model", None)
    try:
        parameter_device = str(next(module.parameters()).device)
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("RF-DETR model does not expose CUDA parameter evidence") from error
    if not declared.casefold().startswith("cuda") or not parameter_device.casefold().startswith(
        "cuda"
    ):
        raise RuntimeError("RF-DETR model is not actually resident on CUDA")
    return {"declared_device": declared, "parameter_device": parameter_device}


def warmup_cuda_model(model, frame) -> dict[str, str]:
    """Trigger RF-DETR's lazy device move, then prove parameters are on CUDA."""
    predict_frame(model, frame, threshold=0.25)
    return cuda_model_evidence(model)


def inspect_video(path: Path) -> dict:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video cannot be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    peak_index = 0
    peak_score = -1.0
    previous = None
    width = height = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (320, 180),
                interpolation=cv2.INTER_AREA,
            )
            if previous is not None:
                score = float(np.mean(cv2.absdiff(gray, previous)))
                if score > peak_score:
                    peak_score = score
                    peak_index = frame_count
            previous = gray
            frame_count += 1
    finally:
        capture.release()
    if frame_count <= 0 or width is None or height is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video metadata or decoding is invalid")
    indices = choose_frame_indices(frame_count, peak_index=peak_index)
    return {
        "decoded_frames": frame_count,
        "fps": fps,
        "height": height,
        "peak_motion_frame": peak_index,
        "peak_motion_score": peak_score,
        "selected_frame_count": len(indices),
        "selected_frame_indices": list(indices),
        "width": width,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def _annotate(frame, detections: list[dict], label: str):
    import cv2

    annotated = frame.copy()
    for item in detections:
        left, top, right, bottom = (int(round(value)) for value in item["bbox_xyxy"])
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            f"{item['confidence']:.2f}",
            (left, max(15, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def infer_clip(model, clip: str, path: Path, plan: dict, output: Path, *, threshold=0.25) -> dict:
    import cv2

    selected = set(int(value) for value in plan["selected_frame_indices"])
    peak = int(plan["peak_motion_frame"])
    representatives = {
        0,
        max(0, peak - 30),
        peak,
        min(int(plan["decoded_frames"]) - 1, peak + 30),
        int(plan["decoded_frames"]) - 1,
    }
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"{clip} video cannot be opened for inference")
    records = []
    timing_ms = []
    confidences = []
    count = 0
    annotations = output / "annotated" / clip
    annotations.mkdir(parents=True, exist_ok=True)
    detections_path = output / f"{clip}-detections.ndjson"
    try:
        with detections_path.open("w", encoding="utf-8") as stream:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = count
                count += 1
                if frame_index not in selected:
                    continue
                started = time.perf_counter_ns()
                detections = predict_frame(model, frame, threshold=threshold)
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                timing_ms.append(elapsed_ms)
                confidences.extend(item["confidence"] for item in detections)
                record = {
                    "clip": clip,
                    "detections": detections,
                    "frame_index": frame_index,
                    "inference_ms": elapsed_ms,
                    "nominal_video_timestamp_ms": (frame_index * 1000.0 / float(plan["fps"])),
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                records.append(record)
                if frame_index in representatives:
                    target = annotations / f"frame-{frame_index:04d}.jpg"
                    annotated = _annotate(frame, detections, f"{clip} frame {frame_index}")
                    if not cv2.imwrite(str(target), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise ValueError("annotated frame could not be encoded")
                if len(records) % 10 == 0:
                    write_json(
                        output / "progress.json",
                        {
                            "clip": clip,
                            "completed_selected_frames": len(records),
                            "updated_ns": time.time_ns(),
                        },
                    )
    finally:
        capture.release()
    if count != int(plan["decoded_frames"]) or len(records) != len(selected):
        raise ValueError(f"{clip} inference did not consume the frozen frame plan")
    detection_counts = [len(record["detections"]) for record in records]
    return {
        "clip": clip,
        "detection_count_maximum": max(detection_counts),
        "detection_count_mean": sum(detection_counts) / len(detection_counts),
        "detection_count_minimum": min(detection_counts),
        "detections_total": sum(detection_counts),
        "frames_with_no_detections": sum(value == 0 for value in detection_counts),
        "inference_ms_median": _percentile(timing_ms, 0.5),
        "inference_ms_p90": _percentile(timing_ms, 0.9),
        "confidence_median": _percentile(confidences, 0.5),
        "confidence_p10": _percentile(confidences, 0.1),
        "confidence_p90": _percentile(confidences, 0.9),
        "sampled_frames": len(records),
    }


def worker_main(arguments) -> int:
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(arguments.video_plan.read_text(encoding="utf-8"))
    model_spec = json.loads(arguments.model_spec)
    verified = verify_model_run(arguments.model_root, model_spec)
    try:
        import torch
        from rfdetr import from_checkpoint

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is intentionally disabled")
        device = {
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_cuda_version": torch.version.cuda,
        }
        started = time.perf_counter()
        model = load_cuda_checkpoint(
            arguments.model_root / "checkpoint_best.pth",
            from_checkpoint,
        )
        load_seconds = time.perf_counter() - started
        import cv2

        first_clip = next(iter(MEDIA_SHA256))
        warmup_capture = cv2.VideoCapture(str(plan["videos"][first_clip]["path"]))
        try:
            if not warmup_capture.isOpened():
                raise RuntimeError("warmup video could not be opened")
            ok, warmup_frame = warmup_capture.read()
            if not ok or warmup_frame is None:
                raise RuntimeError("warmup video did not yield its first frame")
        finally:
            warmup_capture.release()
        device["model_after_unmeasured_warmup"] = warmup_cuda_model(model, warmup_frame)
        torch.cuda.synchronize()
        clips = []
        for clip in MEDIA_SHA256:
            clips.append(
                infer_clip(
                    model,
                    clip,
                    Path(plan["videos"][clip]["path"]),
                    plan["videos"][clip],
                    output,
                )
            )
        summary = {
            "status": "COLLECTED_REVIEW_REQUIRED",
            "label": model_spec["label"],
            "model": verified,
            "device": device,
            "model_load_seconds": load_seconds,
            "runtime_versions": {
                "rfdetr": importlib.metadata.version("rfdetr"),
                "torch": importlib.metadata.version("torch"),
            },
            "clips": clips,
            "identity_evaluated": False,
            "production_ready": False,
        }
        device["model_after_inference"] = cuda_model_evidence(model)
        write_json(output / "worker-summary.json", summary)
        return 0
    except Exception as error:
        write_json(
            output / "worker-summary.json",
            {
                "status": "FAILED_REVIEW_REQUIRED",
                "label": model_spec.get("label"),
                "error_type": type(error).__name__,
                "reason": str(error)[:1000],
                "identity_evaluated": False,
                "production_ready": False,
            },
        )
        return 2


def _run_worker(
    command: list[str],
    output: Path,
    *,
    timeout_seconds=3600,
    stall_seconds=600,
) -> dict:
    def stop_owned_process(process) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    log = output.with_suffix(".log")
    started = time.monotonic()
    last_progress = started
    last_mtime = None
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        try:
            while process.poll() is None:
                now = time.monotonic()
                progress = output / "progress.json"
                try:
                    current_mtime = progress.stat().st_mtime_ns
                    if current_mtime != last_mtime:
                        last_mtime = current_mtime
                        last_progress = now
                except FileNotFoundError:
                    pass
                if now - started > timeout_seconds or now - last_progress > stall_seconds:
                    stop_owned_process(process)
                    return {"status": "TIMEOUT", "exit_code": process.returncode}
                time.sleep(1)
        except BaseException:
            stop_owned_process(process)
            raise
    return {"status": "EXITED", "exit_code": process.returncode}


def pack_result(directory: Path, summary: dict) -> Path:
    directory = Path(directory)
    target = directory.with_name("SEND_BACK-" + directory.name + ".zip")
    if target.exists():
        raise FileExistsError(target)
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if _is_link(path):
            raise ValueError("result contains a linked path")
        if path.is_file():
            suffix = path.suffix.casefold()
            if suffix in {".avi", ".mp4", ".mov", ".mkv"}:
                raise ValueError("result must not contain an original video")
            if suffix in {".onnx", ".pt", ".pth", ".safetensors"}:
                raise ValueError("result must not contain a model weight")
            if path.name != "summary.json":
                inventory[path.relative_to(directory).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
    write_json(
        directory / "summary.json",
        {
            **summary,
            "acceptance": "NOT_EVALUATED",
            "identity_evaluated": False,
            "official_results_written": False,
            "production_ready": False,
            "source_kind": "rfdetr_detector_diagnostic_only",
            "inventory": inventory,
        },
    )
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, directory.name + "/" + path.relative_to(directory).as_posix())
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--downloads-root", type=Path)
    parser.add_argument("--algorithm-root", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--model-spec")
    parser.add_argument("--video-plan", type=Path)
    return parser


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.worker:
        worker_values = (
            arguments.output,
            arguments.model_root,
            arguments.model_spec,
            arguments.video_plan,
        )
        if not all(worker_values):
            raise SystemExit("worker arguments are incomplete")
        return worker_main(arguments)
    if not all((arguments.output, arguments.downloads_root, arguments.algorithm_root)):
        raise SystemExit("--output, --downloads-root and --algorithm-root are required")
    arguments.output.mkdir(parents=True, exist_ok=True)
    unexpected = [
        path for path in arguments.output.iterdir() if path.name != "packet-manifest.json"
    ]
    if unexpected:
        raise ValueError("output directory is not empty")
    resolution = None
    errors = []
    comparison = None
    try:
        resolution = resolve_media(arguments.downloads_root)
        verified_models = []
        for spec in MODEL_SPECS:
            root = arguments.algorithm_root / Path(spec["relative_path"])
            verified_models.append({**spec, **verify_model_run(root, spec), "path": str(root)})
        video_plan = {
            "schema_version": "1.0",
            "source_kind": resolution.source_kind,
            "videos": {},
        }
        for clip, path in resolution.paths.items():
            print(f"Inspecting frozen video: {clip}", flush=True)
            video_plan["videos"][clip] = {
                **inspect_video(path),
                "path": str(path),
                "sha256": MEDIA_SHA256[clip],
            }
        plan_path = arguments.output / "video-plan.json"
        write_json(plan_path, video_plan)
        packet_manifest = Path(__file__).resolve().parent / "packet-manifest.json"
        if packet_manifest.is_file() and not (arguments.output / "packet-manifest.json").exists():
            (arguments.output / "packet-manifest.json").write_bytes(packet_manifest.read_bytes())
        workers = []
        for spec in MODEL_SPECS:
            label = spec["label"]
            worker_output = arguments.output / label
            root = arguments.algorithm_root / Path(spec["relative_path"])
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--output",
                str(worker_output),
                "--model-root",
                str(root),
                "--model-spec",
                json.dumps(spec, separators=(",", ":")),
                "--video-plan",
                str(plan_path),
            ]
            print(f"Running RF-DETR candidate: {label}", flush=True)
            execution = _run_worker(command, worker_output)
            summary_path = worker_output / "worker-summary.json"
            worker_summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file()
                else {"status": "NO_WORKER_SUMMARY"}
            )
            workers.append({"label": label, "execution": execution, "result": worker_summary})
            worker_failed = (
                execution.get("exit_code") != 0
                or worker_summary.get("status") != "COLLECTED_REVIEW_REQUIRED"
            )
            if worker_failed:
                errors.append(f"{label}:WORKER_FAILED")
        comparison = {
            "schema_version": "1.0",
            "packet_version": PACKET_VERSION,
            "baseline_commit": BASELINE,
            "media": {
                name: {key: value for key, value in details.items() if key != "path"}
                for name, details in video_plan["videos"].items()
            },
            "models": verified_models,
            "workers": workers,
            "identity_models_skipped": "all inventoried identity runs have gate_status FAIL",
            "status": "COLLECTED_REVIEW_REQUIRED" if not errors else "PARTIAL_REVIEW_REQUIRED",
        }
        write_json(arguments.output / "comparison.json", comparison)
    except BaseException as error:
        errors.append(f"ORCHESTRATION:{type(error).__name__}")
        comparison = {
            "schema_version": "1.0",
            "packet_version": PACKET_VERSION,
            "baseline_commit": BASELINE,
            "status": "FAILED_REVIEW_REQUIRED",
            "error_type": type(error).__name__,
            "reason": str(error)[:1000],
        }
        write_json(arguments.output / "comparison.json", comparison)
    finally:
        if resolution is not None:
            resolution.close()
    target = pack_result(
        arguments.output,
        {
            "schema_version": "1.0",
            "packet_version": PACKET_VERSION,
            "baseline_commit": BASELINE,
            "status": comparison["status"],
            "errors": errors,
            "models_compared": len(comparison.get("workers", [])),
            "videos_compared": len(comparison.get("media", {})),
        },
    )
    print(json.dumps({"result_zip": str(target), "sha256": sha256(target)}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
