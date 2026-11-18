import cv2
import numpy as np

from billiards_ai.lab.adapters.deadpool_color import DeadpoolColorBackend


def _circle(color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(mask, (32, 32), 24, 1, -1)
    image[mask.astype(bool)] = color
    return image, mask


def _striped_circle(color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _circle(color)
    image[24:40][mask[24:40].astype(bool)] = (245, 245, 245)
    return image, mask


def test_white_mask_resolves_cue_group_without_forcing_number() -> None:
    image, mask = _circle((245, 245, 245))
    scores = DeadpoolColorBackend().predict({"image": image, "mask": mask})
    assert max(scores["group_scores"], key=scores["group_scores"].get) == "cue"
    assert scores["exact_scores"]["unknown"] > scores["exact_scores"]["1"]


def test_striped_crop_prefers_stripe_group() -> None:
    image, mask = _striped_circle(color=(220, 80, 30))
    scores = DeadpoolColorBackend().predict({"image": image, "mask": mask})
    assert scores["group_scores"]["stripe"] > scores["group_scores"]["solid"]


def test_tiny_mask_abstains() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    scores = DeadpoolColorBackend().predict({"image": image, "mask": mask})
    assert scores["exact_scores"]["unknown"] == 1.0
    assert scores["group_scores"]["unknown"] == 1.0


def test_invalid_image_mask_geometry_is_rejected() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    try:
        DeadpoolColorBackend().predict({"image": image, "mask": mask})
    except ValueError as exc:
        assert "same height and width" in str(exc)
    else:
        raise AssertionError("invalid crop geometry was accepted")
