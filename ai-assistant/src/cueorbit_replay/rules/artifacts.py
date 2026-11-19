"""Explicit fixed-revision prefetch; never called by HTTP/startup/encoder."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import httpx

from ..fixtures import ROOT
from .contracts import MODEL_NAME, MODEL_REVISION
from .embeddings import verify_artifacts


def fetch_model(root: Path, *, manifest=None, client=None):
    manifest = manifest or json.loads(
        (ROOT / "data/rules/model-manifest.json").read_text()
    )
    if (manifest["model"], manifest["revision"]) != (MODEL_NAME, MODEL_REVISION):
        raise ValueError("UNSUPPORTED_ENCODER_IDENTITY")
    if root.exists():
        verify_artifacts(root, manifest)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    owned = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=120, trust_env=False)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".verified-model-", dir=root.parent
        ) as staging:
            stage = Path(staging)
            for item in manifest["files"]:
                path = stage / item["file"]
                if not path.resolve().is_relative_to(stage.resolve()):
                    raise ValueError("INVALID_ARTIFACT_PATH")
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                url = f"https://huggingface.co/{MODEL_NAME}/resolve/{MODEL_REVISION}/{item['file']}"
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    if response.url.scheme != "https":
                        raise ValueError("ARTIFACT_HTTPS_REQUIRED")
                    with path.open("wb") as target:
                        for part in response.iter_bytes(1024 * 1024):
                            size += len(part)
                            if size > item["bytes"]:
                                raise ValueError("MODEL_ARTIFACT_SIZE_MISMATCH")
                            digest.update(part)
                            target.write(part)
                if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                    raise ValueError("MODEL_ARTIFACT_HASH_MISMATCH")
            verify_artifacts(stage, manifest)
            os.rename(stage, root)
    finally:
        if owned:
            client.close()
