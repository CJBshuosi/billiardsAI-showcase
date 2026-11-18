"""Runtime identity backends with fixed local assets and lazy optional models."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


class Dinov2AssetError(RuntimeError):
    """Frozen DINOv2 assets are missing, changed, or unsafe."""


EXPECTED_ASSETS: dict[str, tuple[int, str]] = {
    "model.safetensors": (
        88_249_960,
        "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    ),
    "config.json": (
        547,
        "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    ),
    "preprocessor_config.json": (
        436,
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dinov2_assets(asset_root: Path) -> dict[str, Path]:
    root = asset_root.resolve()
    if asset_root.is_symlink() or not root.is_dir():
        raise Dinov2AssetError("DINOv2 asset root is missing or unsafe")
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(EXPECTED_ASSETS):
        raise Dinov2AssetError("DINOv2 asset inventory is invalid")
    verified: dict[str, Path] = {}
    for name, (expected_bytes, expected_sha256) in sorted(EXPECTED_ASSETS.items()):
        path = root / name
        if (
            path.is_symlink()
            or path.stat().st_size != expected_bytes
            or _sha256(path) != expected_sha256
        ):
            raise Dinov2AssetError(f"DINOv2 payload identity changed: {name}")
        verified[name] = path
    return verified


class Dinov2IdentityFeatureBackend:
    """Extract CLS features locally without any Hub or network fallback."""

    version = "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056"

    def __init__(self, *, asset_root: Path, device: str = "cuda", batch_size: int = 32) -> None:
        verify_dinov2_assets(asset_root)
        if batch_size < 1 or batch_size > 256:
            raise ValueError("DINOv2 batch size is invalid")
        self.asset_root = asset_root.resolve()
        self.device = device
        self.batch_size = batch_size
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise Dinov2AssetError("approved DINOv2 runtime dependencies are unavailable") from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise Dinov2AssetError("CUDA is unavailable for DINOv2 feature extraction")
        # Assets were integrity-checked above and network access is disabled for both calls.
        self._processor = AutoImageProcessor.from_pretrained(  # nosec B615
            self.asset_root,
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(  # nosec B615
            self.asset_root,
            local_files_only=True,
            use_safetensors=True,
        ).to(self.device)
        self._model.eval()
        self._torch = torch

    def prepare(self) -> None:
        """Load the same local models once without accepting crops or inferring."""
        self._load()

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray:
        if not crops:
            raise ValueError("DINOv2 feature extraction requires crops")
        self._load()
        from PIL import Image

        outputs = []
        with self._torch.inference_mode():
            for offset in range(0, len(crops), self.batch_size):
                images = []
                for path in crops[offset : offset + self.batch_size]:
                    if path.is_symlink() or not path.is_file():
                        raise Dinov2AssetError("identity crop is missing or unsafe")
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self._processor(images=images, return_tensors="pt")
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                features = self._model(**inputs).last_hidden_state[:, 0]
                outputs.append(features.detach().float().cpu().numpy())
        return np.concatenate(outputs, axis=0)


class OpenCvIdentityVisualEvidenceBackend:
    """Extract color and spatial-pattern evidence from the centred ball ROI.

    The historical name is retained because this evidence mirrors the project's
    OpenCV DEADPOOL baseline. Pillow performs decoding so the training contract
    does not acquire another runtime dependency.
    """

    version = "deadpool-centred-ball-evidence/1"
    feature_names = (
        "white_ratio",
        "black_ratio",
        "color_ratio",
        "saturation_mean",
        "value_mean",
        "white_spatial_std",
        "color_spatial_std",
        "radial_saturation_contrast",
    )

    def __init__(self, *, context_scale: float = 2.0) -> None:
        if not math.isfinite(context_scale) or not 1 <= context_scale <= 4:
            raise ValueError("identity evidence context scale is invalid")
        self.context_scale = context_scale

    @staticmethod
    def _decode(path: Path) -> np.ndarray:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise RuntimeError("approved Pillow dependency is unavailable") from exc
        try:
            with Image.open(path) as source:
                return np.asarray(source.convert("HSV"), dtype=np.uint8)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("identity evidence crop cannot be decoded") from exc

    def _extract_one(self, path: Path) -> np.ndarray:
        hsv = self._decode(path)
        height, width = hsv.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius = min(width, height) / (2.0 * self.context_scale)
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        selected = distance <= radius
        if int(selected.sum()) < 16:
            raise ValueError("identity evidence ball ROI is too small")
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        white = (saturation < 35) & (value > 150)
        black = value < 55
        color = (saturation >= 55) & (value >= 70)

        angles = (np.arctan2(yy - center_y, xx - center_x) + 2 * np.pi) % (2 * np.pi)
        sectors = np.floor(angles / (2 * np.pi / 8)).astype(np.int64)

        def spatial_std(mask: np.ndarray) -> float:
            ratios = [float(np.mean(mask[selected & (sectors == sector)])) for sector in range(8)]
            return min(1.0, float(np.std(ratios)) * 2.0)

        inner = selected & (distance <= radius * 0.55)
        outer = selected & ~inner
        radial_contrast = (
            abs(float(np.mean(saturation[inner])) - float(np.mean(saturation[outer]))) / 255.0
        )
        return np.asarray(
            (
                np.mean(white[selected]),
                np.mean(black[selected]),
                np.mean(color[selected]),
                np.mean(saturation[selected]) / 255.0,
                np.mean(value[selected]) / 255.0,
                spatial_std(white),
                spatial_std(color),
                radial_contrast,
            ),
            dtype=np.float32,
        )

    def extract(self, crops: tuple[Path, ...]) -> np.ndarray:
        if not crops:
            raise ValueError("identity evidence requires crops")
        return np.stack([self._extract_one(path) for path in crops])
