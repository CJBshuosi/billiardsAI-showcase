from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from billiards_ai.lab.training.identity_visual_evidence import (
    OpenCvIdentityVisualEvidenceBackend,
)


def _ball_crop(path: Path, *, kind: str) -> None:
    image = Image.new("RGB", (96, 96), (25, 130, 25))
    draw = ImageDraw.Draw(image)
    ball = (24, 24, 72, 72)
    if kind == "cue":
        draw.ellipse(ball, fill=(235, 235, 235))
    elif kind == "eight":
        draw.ellipse(ball, fill=(15, 15, 15))
    elif kind == "solid":
        draw.ellipse(ball, fill=(220, 20, 20))
    elif kind == "stripe":
        draw.ellipse(ball, fill=(235, 235, 235))
        draw.rectangle((25, 40, 71, 56), fill=(220, 20, 20))
        draw.ellipse(ball, outline=(0, 0, 0), width=1)
    else:
        raise AssertionError(kind)
    image.save(path, format="PNG")


def test_visual_evidence_separates_white_black_solid_and_stripe_crops(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"{kind}.png" for kind in ("cue", "eight", "solid", "stripe"))
    for path in paths:
        _ball_crop(path, kind=path.stem)

    backend = OpenCvIdentityVisualEvidenceBackend(context_scale=2.0)
    features = backend.extract(paths)
    columns = {name: index for index, name in enumerate(backend.feature_names)}

    assert features.shape == (4, len(backend.feature_names))
    assert features[0, columns["white_ratio"]] > 0.90
    assert features[1, columns["black_ratio"]] > 0.90
    assert features[2, columns["color_ratio"]] > 0.90
    assert 0.20 < features[3, columns["white_ratio"]] < 0.80
    assert 0.20 < features[3, columns["color_ratio"]] < 0.80
    assert features[3, columns["color_spatial_std"]] > features[2, columns["color_spatial_std"]]


def test_visual_evidence_rejects_undecodable_crop(tmp_path: Path) -> None:
    crop = tmp_path / "broken.png"
    crop.write_bytes(b"not-an-image")

    backend = OpenCvIdentityVisualEvidenceBackend(context_scale=2.0)

    try:
        backend.extract((crop,))
    except ValueError as exc:
        assert str(exc) == "identity evidence crop cannot be decoded"
    else:
        raise AssertionError("undecodable crop was accepted")
