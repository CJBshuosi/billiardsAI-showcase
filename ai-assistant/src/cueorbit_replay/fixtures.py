from pathlib import Path

from .contracts import ReplayBundleV1

ROOT = Path(__file__).resolve().parents[2]


def load_bundle(path: Path) -> ReplayBundleV1:
    return ReplayBundleV1.model_validate_json(path.read_text())


def load_builtin_bundles() -> tuple[ReplayBundleV1, ...]:
    return tuple(
        load_bundle(ROOT / "data" / "sessions" / f"{name}.json")
        for name in ("complete", "partial", "empty")
    )
