"""Pinned local encoder; optional native imports occur only at construction."""

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

from ..fixtures import ROOT
from .contracts import MODEL_NAME, MODEL_REVISION


class EmbeddingPort(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def validate_vector(vector) -> list[float]:
    values = [float(v) for v in vector]
    if len(values) != 384 or not all(math.isfinite(v) for v in values):
        raise ValueError("INVALID_VECTOR")
    if not math.isclose(sum(v * v for v in values), 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("VECTOR_NOT_NORMALIZED")
    return values


def verify_artifacts(root: Path, manifest: dict):
    root = root.resolve()
    for item in manifest["files"]:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("MODEL_ARTIFACT_MISSING")
        if (
            path.stat().st_size != item["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("MODEL_ARTIFACT_HASH_MISMATCH")


class LocalEncoder:
    def __init__(self, root: Path):
        manifest = json.loads((ROOT / "data/rules/model-manifest.json").read_text())
        if manifest["model"] != MODEL_NAME or manifest["revision"] != MODEL_REVISION:
            raise ValueError("UNSUPPORTED_ENCODER_IDENTITY")
        verify_artifacts(root, manifest)
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        if not any(
            m["model"] == MODEL_NAME for m in TextEmbedding.list_supported_models()
        ):
            TextEmbedding.add_custom_model(
                model=MODEL_NAME,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=MODEL_NAME),
                dim=384,
                model_file="onnx/model.onnx",
            )
        self.model = TextEmbedding(
            model_name=MODEL_NAME,
            specific_model_path=str(root.resolve()),
            local_files_only=True,
            cache_dir=str(root.parent / "fastembed-cache"),
            providers=["CPUExecutionProvider"],
            threads=2,
        )

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        inputs = [prefix + text for text in texts]
        if any(self.count(t) > 512 for t in inputs):
            raise ValueError("ENCODER_TOKEN_LIMIT")
        return [validate_vector(v) for v in self.model.embed(inputs, batch_size=8)]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query: ")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "passage: ")
