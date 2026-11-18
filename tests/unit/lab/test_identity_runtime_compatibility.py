"""Runtime preprocessing preserves captured pixel and numerical contracts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util

import numpy as np
import pytest


def canonical(name):
    path = "billiards_ai.lab.mainline1_identity_" + name
    assert importlib.util.find_spec(path) is not None, "canonical identity module is missing"
    return importlib.import_module(path)


@pytest.mark.parametrize(
    "name",
    [
        "Dinov2AssetError",
        "EXPECTED_ASSETS",
        "verify_dinov2_assets",
        "Dinov2IdentityFeatureBackend",
        "_sha256",
    ],
)
def test_legacy_dinov2_exports_share_one_runtime_implementation(name):
    backend = canonical("backends")
    old = importlib.import_module("billiards_ai.lab.training.identity_dinov2")
    assert getattr(old, name) is getattr(backend, name)


def test_visual_evidence_old_and_runtime_paths_share_one_class():
    backend = canonical("backends")
    old = importlib.import_module("billiards_ai.lab.training.identity_visual_evidence")
    assert old.OpenCvIdentityVisualEvidenceBackend is backend.OpenCvIdentityVisualEvidenceBackend


def test_runtime_and_training_share_crop_normalisation_and_exception_objects():
    prep = canonical("preprocessing")
    crop = importlib.import_module("billiards_ai.lab.training.identity_crop_export")
    group = importlib.import_module("billiards_ai.lab.training.identity_group_backend")
    runtime = importlib.import_module("billiards_ai.lab.mainline1_runtime")
    assert crop.IdentityCropExportError is prep.IdentityCropExportError
    assert crop._square_crop is runtime._square_crop is prep.square_identity_crop
    assert group._normalise is runtime._normalise is prep.normalise_identity_features


@pytest.mark.parametrize(
    "box,scale,size,digest",
    [
        (
            (-1.5, -2.0, 4.0, 3.0),
            2.0,
            8,
            "70d690e2bef8ba6d5232c0fa94194f102ab3b3d0835063fa1fab4413f13de8b4",
        ),
        (
            (1.0, 1.0, 3.0, 3.0),
            1.0,
            4,
            "ab750611af9f2c3577f5ca4532246838a31f6dfcf99f7de0b6504a9a8ab034ec",
        ),
    ],
)
def test_crop_keeps_pre_move_border_replication_and_cubic_pixels(box, scale, size, digest):
    prep = canonical("preprocessing")
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    result = prep.square_identity_crop(image, box, context_scale=scale, output_size=size)
    assert result.dtype == np.uint8 and result.shape == (size, size, 3)
    assert hashlib.sha256(result.tobytes()).hexdigest() == digest


def test_normalisation_keeps_float32_zero_and_tiny_vector_floor():
    prep = canonical("preprocessing")
    features = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1e-20, 0.0, -1e-20]])
    result = prep.normalise_identity_features(features)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result,
        np.asarray([[0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [1e-8, 0.0, -1e-8]], dtype=np.float32),
    )


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2,)])
def test_normalisation_still_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError, match="DINOv2 features are invalid"):
        canonical("preprocessing").normalise_identity_features(np.empty(shape))


def test_outside_crop_raises_the_original_compatible_exception():
    prep = canonical("preprocessing")
    old = importlib.import_module("billiards_ai.lab.training.identity_crop_export")
    with pytest.raises(old.IdentityCropExportError, match="outside its source image"):
        prep.square_identity_crop(
            np.zeros((5, 7, 3), dtype=np.uint8),
            (100.0, 100.0, 2.0, 2.0),
            context_scale=1.0,
            output_size=4,
        )
