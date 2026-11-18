"""Run the real pooltool API in its dependency-isolated environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def run_native_smoke(pooltool: Any) -> dict[str, object]:
    balls = {
        "cue": pooltool.Ball.create("cue", xy=(0.5, 0.6)),
        "target": pooltool.Ball.create("target", xy=(1.5, 0.6)),
    }
    cue = pooltool.Cue(cue_ball_id="cue")
    cue.set_state(V0=1.5, phi=0.0)
    system = pooltool.System(cue=cue, table=pooltool.Table.default(), balls=balls)
    simulated = pooltool.simulate(system, continuous=True, dt=1 / 60)
    if simulated is not None:
        system = simulated
    events = tuple(getattr(system, "events", ()))
    if len(getattr(system, "balls", {})) != 2 or not events:
        raise RuntimeError("pooltool native simulation returned incomplete state")
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "pooltool_version": str(getattr(pooltool, "__version__", "unknown")),
        "ball_count": len(system.balls),
        "event_count": len(events),
        "synthetic_only": True,
    }


def publish(document: dict[str, object], output_root: Path) -> Path:
    payload = _canonical(document)
    identity = hashlib.sha256(payload).hexdigest()
    destination = output_root.resolve() / identity
    if destination.exists():
        if (destination / "result.json").read_bytes() == payload:
            return destination
        raise RuntimeError("pooltool native smoke artifact conflicts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{identity}.tmp-{uuid.uuid4()}"
    temporary.mkdir()
    try:
        (temporary / "result.json").write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.rmdir()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    import pooltool

    destination = publish(run_native_smoke(pooltool), arguments.output_root)
    print(json.dumps({"status": "PASS", "output": str(destination)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
