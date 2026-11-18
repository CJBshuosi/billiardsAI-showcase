from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

from billiards_ai.lab.training.detector_dataset import create_synthetic_rfdetr_fixture


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _decode_rgb_png(payload: bytes) -> tuple[int, int, bytes]:
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_data = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            assert (depth, color_type, compression, filtering, interlace) == (8, 2, 0, 0, 0)
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    scanlines = zlib.decompress(bytes(compressed))
    row_bytes = width * 3
    assert len(scanlines) == height * (row_bytes + 1)
    assert all(scanlines[row * (row_bytes + 1)] == 0 for row in range(height))
    pixels = b"".join(
        scanlines[row * (row_bytes + 1) + 1 : (row + 1) * (row_bytes + 1)] for row in range(height)
    )
    return width, height, pixels


def test_synthetic_rfdetr_fixture_is_deterministic_decodable_and_coco_valid(
    tmp_path: Path,
) -> None:
    first = create_synthetic_rfdetr_fixture(tmp_path / "first")
    second = create_synthetic_rfdetr_fixture(tmp_path / "second")

    assert first.data_hash == second.data_hash
    assert _tree_hashes(first.output) == _tree_hashes(second.output)
    image_path = first.output / "train" / "frame-1.png"
    width, height, pixels = _decode_rgb_png(image_path.read_bytes())
    assert (width, height) == (64, 64)
    assert len(pixels) == 64 * 64 * 3
    for split in ("train", "valid"):
        annotation_path = first.output / split / "_annotations.coco.json"
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        assert coco["images"] == [{"file_name": "frame-1.png", "height": 64, "id": 1, "width": 64}]
        assert coco["annotations"][0]["segmentation"] == [[24, 24, 40, 24, 40, 40, 24, 40]]
        assert (first.output / split / coco["images"][0]["file_name"]).is_file()
