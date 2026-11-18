"""Command-line entry point for the local algorithm laboratory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import yaml

from billiards_ai.lab.benchmark import MetricResult, run_benchmark
from billiards_ai.lab.dataset import freeze_dataset
from billiards_ai.lab.failures import (
    FailureDimensions,
    FailureRecord,
    FailureSeverity,
    build_failure_matrix,
)
from billiards_ai.lab.render import build_overlay_frames
from billiards_ai.lab.runner import RunInputs, create_artifact_run


class _ExistingOverlayRenderer:
    name = "existing-overlay"

    def __init__(self, source: Path) -> None:
        self.source = source
        self.version = hashlib.sha256(source.read_bytes()).hexdigest()

    def render(self, *, output_path: Path, overlay_frames: tuple[dict, ...]) -> None:
        del overlay_frames
        shutil.copyfile(self.source, output_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="billiards-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    freeze = dataset_commands.add_parser("freeze")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    ingest = commands.add_parser("ingest", help="freeze an approved local dataset")
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="assemble an immutable artifact run")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--runs-root", type=Path, required=True)
    run.add_argument("--overlay-source", type=Path, required=True)

    benchmark = commands.add_parser("benchmark", help="evaluate registered hard gates")
    benchmark.add_argument("--metrics", type=Path, required=True)
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--prediction-hash", action="append", required=True)
    benchmark.add_argument("--output", type=Path, required=True)

    render = commands.add_parser("render", help="build renderer-neutral overlay instructions")
    render.add_argument("--predictions", type=Path, required=True)
    render.add_argument("--evidence", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)

    failures = commands.add_parser("failures", help="aggregate the failure matrix")
    failures.add_argument("--source", type=Path, required=True)
    failures.add_argument("--output", type=Path, required=True)

    classic = commands.add_parser(
        "classic-cv-synthetic-baseline",
        help="run the decoded-PNG classic-CV synthetic regression",
    )
    classic.add_argument("--output", type=Path, required=True)
    return parser


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.command == "dataset" and args.dataset_command == "freeze") or args.command == "ingest":
        manifest = freeze_dataset(args.source, args.output)
        print(
            json.dumps(
                {
                    "revision": manifest.revision,
                    "locked": manifest.locked,
                    "output": str(args.output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        inputs = RunInputs.model_validate(_load_json(args.inputs))
        run_dir = create_artifact_run(
            args.runs_root,
            inputs,
            _ExistingOverlayRenderer(args.overlay_source),
        )
        print(json.dumps({"run_dir": str(run_dir)}, sort_keys=True))
        return 0
    if args.command == "benchmark":
        metric_rows = _load_json(args.metrics)
        if not isinstance(metric_rows, list):
            raise ValueError("metrics input must be a JSON list")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        result = run_benchmark(
            metrics=[MetricResult(**row) for row in metric_rows],
            config=config,
            repeated_prediction_hashes=tuple(args.prediction_hash),
        )
        _write_json(args.output, asdict(result))
        return 0
    if args.command == "render":
        predictions = _load_json(args.predictions)
        evidence = _load_json(args.evidence)
        if not isinstance(predictions, list) or not isinstance(evidence, dict):
            raise ValueError("render inputs must be predictions list and evidence object")
        frames = build_overlay_frames(
            predictions,
            {int(frame_id): frame for frame_id, frame in evidence.items()},
        )
        _write_json(args.output, frames)
        return 0
    if args.command == "failures":
        rows = _load_json(args.source)
        if not isinstance(rows, list):
            raise ValueError("failure input must be a JSON list")
        records = [
            FailureRecord(
                shot_id=row["shot_id"],
                stage=row["stage"],
                field=row["field"],
                severity=FailureSeverity(row["severity"]),
                confidence=row["confidence"],
                dimensions=FailureDimensions(**row["dimensions"]),
            )
            for row in rows
        ]
        _write_json(args.output, [asdict(bucket) for bucket in build_failure_matrix(records)])
        return 0
    if args.command == "classic-cv-synthetic-baseline":
        from billiards_ai.lab.adapters.classic_cv import run_synthetic_classic_cv_baseline

        evidence = run_synthetic_classic_cv_baseline()
        _write_json(args.output, evidence)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": evidence["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
