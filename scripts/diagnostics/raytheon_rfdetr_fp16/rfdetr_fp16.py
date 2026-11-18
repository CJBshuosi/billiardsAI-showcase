"""Benchmark verified RF-DETR candidates in FP32 and FP16 on frozen RER videos.

This is an offline detector-only diagnostic. It does not access a camera, field
service, identity model, calibration store, match state, or official scores.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import rfdetr_compare as base  # noqa: E402
except ModuleNotFoundError:
    _base_path = (
        Path(__file__).resolve().parents[1] / "raytheon_rfdetr_compare" / "rfdetr_compare.py"
    )
    _base_spec = importlib.util.spec_from_file_location("rfdetr_compare", _base_path)
    base = importlib.util.module_from_spec(_base_spec)
    sys.modules[_base_spec.name] = base
    _base_spec.loader.exec_module(base)

PACKET_VERSION = "raytheon-rfdetr-fp16-20260906-v1"
BASELINE = base.BASELINE
PRECISION_MODES = ("fp32", "fp16")


def _box_iou(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def compare_detection_sets(
    baseline: list[dict], candidate: list[dict], *, minimum_iou: float = 0.5
) -> dict:
    """Greedily match same-class detections by descending IoU."""
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be between zero and one")
    choices = []
    for baseline_index, baseline_item in enumerate(baseline):
        for candidate_index, candidate_item in enumerate(candidate):
            if baseline_item["class_id"] != candidate_item["class_id"]:
                continue
            iou = _box_iou(baseline_item["bbox_xyxy"], candidate_item["bbox_xyxy"])
            if iou >= minimum_iou:
                choices.append((iou, baseline_index, candidate_index))
    used_baseline = set()
    used_candidate = set()
    matched_iou = []
    confidence_delta = []
    for iou, baseline_index, candidate_index in sorted(choices, reverse=True):
        if baseline_index in used_baseline or candidate_index in used_candidate:
            continue
        used_baseline.add(baseline_index)
        used_candidate.add(candidate_index)
        matched_iou.append(iou)
        confidence_delta.append(
            candidate[candidate_index]["confidence"] - baseline[baseline_index]["confidence"]
        )
    return {
        "baseline_count": len(baseline),
        "candidate_count": len(candidate),
        "matched_count": len(matched_iou),
        "unmatched_baseline_count": len(baseline) - len(matched_iou),
        "unmatched_candidate_count": len(candidate) - len(matched_iou),
        "matched_iou": matched_iou,
        "matched_confidence_delta": confidence_delta,
    }


def _active_module(model):
    context = getattr(model, "model", None)
    optimized = getattr(context, "inference_model", None)
    return optimized if optimized is not None else getattr(context, "model", None)


def cuda_model_evidence(model, *, expected_dtype: str | None = None) -> dict[str, str]:
    context = getattr(model, "model", None)
    declared = str(getattr(context, "device", ""))
    module = _active_module(model)
    try:
        parameter = next(module.parameters())
        parameter_device = str(parameter.device)
        parameter_dtype = str(parameter.dtype)
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("RF-DETR model does not expose CUDA dtype evidence") from error
    if not declared.casefold().startswith("cuda") or not parameter_device.casefold().startswith(
        "cuda"
    ):
        raise RuntimeError("RF-DETR model is not actually resident on CUDA")
    if expected_dtype is not None and parameter_dtype.casefold() != expected_dtype.casefold():
        raise RuntimeError(
            f"RF-DETR optimized parameter dtype is {parameter_dtype}, expected {expected_dtype}"
        )
    return {
        "declared_device": declared,
        "parameter_device": parameter_device,
        "parameter_dtype": parameter_dtype,
    }


def enable_fp16(model, torch_module) -> dict[str, str]:
    """Use RF-DETR's public in-place inference optimizer and prove FP16 residency."""
    model.inference(compile=False, inplace=True, dtype=torch_module.float16)
    return cuda_model_evidence(model, expected_dtype="torch.float16")


def timed_predict(model, frame, *, threshold: float, cuda) -> tuple[list[dict], float]:
    cuda.synchronize()
    started = time.perf_counter_ns()
    detections = base.predict_frame(model, frame, threshold=threshold)
    cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return detections, elapsed_ms


def _percentile(values: list[float], percentile: float) -> float | None:
    return base._percentile(values, percentile)


def _timing_summary(values: list[float]) -> dict:
    median = _percentile(values, 0.5)
    return {
        "inference_ms_median": median,
        "inference_ms_p90": _percentile(values, 0.9),
        "effective_fps_from_median": 1000.0 / median if median and median > 0 else None,
        "sampled_frames": len(values),
    }


def summarize_precision_comparison(
    *, fp32_timings: list[float], fp16_timings: list[float], comparisons: list[dict]
) -> dict:
    matched_iou = [value for item in comparisons for value in item["matched_iou"]]
    confidence_delta = [value for item in comparisons for value in item["matched_confidence_delta"]]
    matched = sum(item["matched_count"] for item in comparisons)
    baseline_total = sum(item["baseline_count"] for item in comparisons)
    candidate_total = sum(item["candidate_count"] for item in comparisons)
    count_agreement = sum(item["baseline_count"] == item["candidate_count"] for item in comparisons)
    fp32 = _timing_summary(fp32_timings)
    fp16 = _timing_summary(fp16_timings)
    fp16_median = fp16["inference_ms_median"]
    fp16_p90 = fp16["inference_ms_p90"]
    budget_30fps = 1000.0 / 30.0
    budget_60fps = 1000.0 / 60.0
    return {
        "fp32": fp32,
        "fp16": fp16,
        "fp16_speedup_from_median": (
            fp32["inference_ms_median"] / fp16_median
            if fp32["inference_ms_median"] and fp16_median
            else None
        ),
        "compared_frames": len(comparisons),
        "count_agreement_frames": count_agreement,
        "count_agreement_rate": count_agreement / len(comparisons) if comparisons else None,
        "matched_count": matched,
        "baseline_detection_count": baseline_total,
        "fp16_detection_count": candidate_total,
        "matched_iou_median": _percentile(matched_iou, 0.5),
        "matched_iou_p10": _percentile(matched_iou, 0.1),
        "matched_confidence_delta_median": _percentile(confidence_delta, 0.5),
        "unmatched_fp32_count": baseline_total - matched,
        "unmatched_fp16_count": candidate_total - matched,
        "frame_budget": {
            "30fps_ms": budget_30fps,
            "60fps_ms": budget_60fps,
            "fp16_median_within_30fps_detector_budget": bool(
                fp16_median and fp16_median <= budget_30fps
            ),
            "fp16_p90_within_30fps_detector_budget": bool(fp16_p90 and fp16_p90 <= budget_30fps),
            "fp16_frames_within_30fps_budget_rate": (
                sum(value <= budget_30fps for value in fp16_timings) / len(fp16_timings)
                if fp16_timings
                else None
            ),
            "fp16_median_within_60fps_detector_budget": bool(
                fp16_median and fp16_median <= budget_60fps
            ),
            "fp16_p90_within_60fps_detector_budget": bool(fp16_p90 and fp16_p90 <= budget_60fps),
            "fp16_frames_within_60fps_budget_rate": (
                sum(value <= budget_60fps for value in fp16_timings) / len(fp16_timings)
                if fp16_timings
                else None
            ),
            "sustained_realtime_acceptance": "NOT_EVALUATED",
        },
        "acceptance": "NOT_EVALUATED",
        "production_ready": False,
    }


def _selected_frames(path: Path, plan: dict):
    import cv2

    selected = set(int(value) for value in plan["selected_frame_indices"])
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video cannot be opened for precision inference")
    frame_index = 0
    delivered = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in selected:
                delivered += 1
                yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()
    if frame_index != int(plan["decoded_frames"]) or delivered != len(selected):
        raise ValueError("precision inference did not consume the frozen frame plan")


def _run_precision_mode(
    model,
    *,
    mode: str,
    plan: dict,
    output: Path,
    cuda,
    threshold: float = 0.25,
) -> tuple[dict[tuple[str, int], list[dict]], list[float], list[dict]]:
    records_by_frame = {}
    timings = []
    clip_summaries = []
    for clip in base.MEDIA_SHA256:
        records = []
        detections_path = output / f"{clip}-{mode}-detections.ndjson"
        with detections_path.open("w", encoding="utf-8") as stream:
            for frame_index, frame in _selected_frames(
                Path(plan["videos"][clip]["path"]), plan["videos"][clip]
            ):
                detections, elapsed_ms = timed_predict(model, frame, threshold=threshold, cuda=cuda)
                timings.append(elapsed_ms)
                records_by_frame[(clip, frame_index)] = detections
                record = {
                    "clip": clip,
                    "frame_index": frame_index,
                    "precision": mode,
                    "detections": detections,
                    "inference_ms": elapsed_ms,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                records.append(record)
                if len(records) % 10 == 0:
                    base.write_json(
                        output / "progress.json",
                        {
                            "clip": clip,
                            "precision": mode,
                            "completed_selected_frames": len(records),
                            "updated_ns": time.time_ns(),
                        },
                    )
        counts = [len(item["detections"]) for item in records]
        clip_summaries.append(
            {
                "clip": clip,
                "precision": mode,
                "sampled_frames": len(records),
                "detections_total": sum(counts),
                "detection_count_minimum": min(counts),
                "detection_count_maximum": max(counts),
            }
        )
    return records_by_frame, timings, clip_summaries


def _memory_evidence(torch_module) -> dict:
    return {
        "max_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
        "memory_allocated_bytes": int(torch_module.cuda.memory_allocated()),
        "memory_reserved_bytes": int(torch_module.cuda.memory_reserved()),
    }


def worker_main(arguments) -> int:
    arguments.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(arguments.video_plan.read_text(encoding="utf-8"))
    model_spec = json.loads(arguments.model_spec)
    verified = base.verify_model_run(arguments.model_root, model_spec)
    try:
        import cv2
        import torch
        from rfdetr import from_checkpoint

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is intentionally disabled")
        model = base.load_cuda_checkpoint(
            arguments.model_root / "checkpoint_best.pth", from_checkpoint
        )
        first_clip = next(iter(base.MEDIA_SHA256))
        warmup_capture = cv2.VideoCapture(str(plan["videos"][first_clip]["path"]))
        try:
            ok, warmup_frame = warmup_capture.read()
            if not warmup_capture.isOpened() or not ok or warmup_frame is None:
                raise RuntimeError("warmup video did not yield its first frame")
        finally:
            warmup_capture.release()
        base.warmup_cuda_model(model, warmup_frame)
        fp32_device = cuda_model_evidence(model, expected_dtype="torch.float32")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fp32_records, fp32_timings, fp32_clips = _run_precision_mode(
            model,
            mode="fp32",
            plan=plan,
            output=arguments.output,
            cuda=torch.cuda,
        )
        fp32_memory = _memory_evidence(torch)

        fp16_device = enable_fp16(model, torch)
        base.predict_frame(model, warmup_frame, threshold=0.25)
        torch.cuda.synchronize()
        fp16_device = cuda_model_evidence(model, expected_dtype="torch.float16")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fp16_records, fp16_timings, fp16_clips = _run_precision_mode(
            model,
            mode="fp16",
            plan=plan,
            output=arguments.output,
            cuda=torch.cuda,
        )
        fp16_memory = _memory_evidence(torch)
        if fp32_records.keys() != fp16_records.keys():
            raise RuntimeError("FP32 and FP16 did not process identical frozen frames")
        comparisons = [
            {
                "clip": key[0],
                "frame_index": key[1],
                **compare_detection_sets(fp32_records[key], fp16_records[key]),
            }
            for key in sorted(fp32_records)
        ]
        with (arguments.output / "fp32-vs-fp16.ndjson").open("w", encoding="utf-8") as stream:
            for item in comparisons:
                stream.write(json.dumps(item, separators=(",", ":")) + "\n")
        summary = {
            "status": "COLLECTED_REVIEW_REQUIRED",
            "label": model_spec["label"],
            "model": verified,
            "runtime_versions": {
                "rfdetr": importlib.metadata.version("rfdetr"),
                "torch": importlib.metadata.version("torch"),
            },
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_cuda_version": torch.version.cuda,
            "device": {"fp32": fp32_device, "fp16": fp16_device},
            "memory": {
                "measurement_scope": (
                    "steady-state after warmup and empty_cache; FP32 then FP16 in one process; "
                    "not independent model residency"
                ),
                "fp32": fp32_memory,
                "fp16": fp16_memory,
            },
            "clips": {"fp32": fp32_clips, "fp16": fp16_clips},
            "comparison": summarize_precision_comparison(
                fp32_timings=fp32_timings,
                fp16_timings=fp16_timings,
                comparisons=comparisons,
            ),
            "identity_evaluated": False,
            "official_results_written": False,
            "production_ready": False,
        }
        base.write_json(arguments.output / "worker-summary.json", summary)
        return 0
    except Exception as error:
        base.write_json(
            arguments.output / "worker-summary.json",
            {
                "status": "FAILED_REVIEW_REQUIRED",
                "label": model_spec.get("label"),
                "error_type": type(error).__name__,
                "reason": str(error)[:1000],
                "identity_evaluated": False,
                "official_results_written": False,
                "production_ready": False,
            },
        )
        return 2


def pack_result(directory: Path, summary: dict) -> Path:
    directory = Path(directory)
    target = directory.with_name("SEND_BACK-" + directory.name + ".zip")
    if target.exists():
        raise FileExistsError(target)
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if base._is_link(path):
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
                    "sha256": base.sha256(path),
                }
    base.write_json(
        directory / "summary.json",
        {
            **summary,
            "acceptance": "NOT_EVALUATED",
            "identity_evaluated": False,
            "official_results_written": False,
            "production_ready": False,
            "source_kind": "rfdetr_fp32_fp16_detector_diagnostic_only",
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
        if not all(
            (
                arguments.output,
                arguments.model_root,
                arguments.model_spec,
                arguments.video_plan,
            )
        ):
            raise SystemExit("worker arguments are incomplete")
        return worker_main(arguments)
    if not all((arguments.output, arguments.downloads_root, arguments.algorithm_root)):
        raise SystemExit("--output, --downloads-root and --algorithm-root are required")
    arguments.output.mkdir(parents=True, exist_ok=True)
    if any(path.name != "packet-manifest.json" for path in arguments.output.iterdir()):
        raise ValueError("output directory is not empty")
    resolution = None
    errors = []
    workers = []
    try:
        resolution = base.resolve_media(arguments.downloads_root)
        verified_models = []
        for spec in base.MODEL_SPECS:
            root = arguments.algorithm_root / Path(spec["relative_path"])
            verified_models.append({**spec, **base.verify_model_run(root, spec), "path": str(root)})
        video_plan = {"schema_version": "1.0", "source_kind": resolution.source_kind, "videos": {}}
        for clip, path in resolution.paths.items():
            print(f"Inspecting frozen video: {clip}", flush=True)
            video_plan["videos"][clip] = {
                **base.inspect_video(path),
                "path": str(path),
                "sha256": base.MEDIA_SHA256[clip],
            }
        plan_path = arguments.output / "video-plan.json"
        base.write_json(plan_path, video_plan)
        packet_manifest = Path(__file__).resolve().parent / "packet-manifest.json"
        if packet_manifest.is_file() and not (arguments.output / "packet-manifest.json").exists():
            (arguments.output / "packet-manifest.json").write_bytes(packet_manifest.read_bytes())
        for spec in base.MODEL_SPECS:
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
            print(f"Running FP32 -> FP16 diagnostic: {label}", flush=True)
            execution = base._run_worker(command, worker_output, timeout_seconds=5400)
            worker_summary_path = worker_output / "worker-summary.json"
            worker_summary = (
                json.loads(worker_summary_path.read_text(encoding="utf-8"))
                if worker_summary_path.is_file()
                else {"status": "NO_WORKER_SUMMARY"}
            )
            workers.append({"label": label, "execution": execution, "result": worker_summary})
            if (
                execution.get("exit_code") != 0
                or worker_summary.get("status") != "COLLECTED_REVIEW_REQUIRED"
            ):
                errors.append(f"{label}:WORKER_FAILED")
        comparison = {
            "schema_version": "1.0",
            "packet_version": PACKET_VERSION,
            "baseline_commit": BASELINE,
            "precision_modes": list(PRECISION_MODES),
            "models": verified_models,
            "workers": workers,
            "identity_models_skipped": "all inventoried identity runs have gate_status FAIL",
            "status": "COLLECTED_REVIEW_REQUIRED" if not errors else "PARTIAL_REVIEW_REQUIRED",
        }
        base.write_json(arguments.output / "comparison.json", comparison)
    except BaseException as error:
        errors.append(f"{type(error).__name__}:{str(error)[:500]}")
        base.write_json(
            arguments.output / "diagnostic-error.json",
            {"error_type": type(error).__name__, "reason": str(error)[:1000]},
        )
    finally:
        if resolution is not None:
            resolution.close()
    result = pack_result(
        arguments.output,
        {
            "status": "COLLECTED_REVIEW_REQUIRED" if not errors else "PARTIAL_REVIEW_REQUIRED",
            "errors": errors,
            "packet_version": PACKET_VERSION,
            "baseline_commit": BASELINE,
            "precision_modes": list(PRECISION_MODES),
            "worker_count": len(workers),
        },
    )
    print(f"SEND_BACK_ZIP={result}", flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
