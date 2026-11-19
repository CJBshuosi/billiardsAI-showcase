"""Developer-only corpus operations. No arbitrary URL/path/import endpoint."""

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone

from ..fixtures import ROOT
from .contracts import PDF_SHA256, RuleSourceManifestV1
from .corpus import extract_pdf
from .embeddings import LocalEncoder
from .source import fetch_source
from .storage import RuleStore

LOCAL = ROOT / "data/rules/local"
MODEL_ROOT = ROOT / "work/models/multilingual-e5-small"
MANIFEST_PATH = ROOT / "data/rules/source-manifest.json"


def source_manifest():
    if MANIFEST_PATH.exists():
        return RuleSourceManifestV1.model_validate_json(MANIFEST_PATH.read_text())
    return RuleSourceManifestV1(
        source_id="wpa-heyball-20250816",
        ruleset_id="wpa-heyball",
        effective_date="2025-08-16",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        pdf_sha256=PDF_SHA256,
        extraction_version="pypdf-7.1.0-nfkc-ws-v2",
    )


async def build_index():
    manifest = source_manifest()
    encoder = LocalEncoder(MODEL_ROOT)
    chunks = extract_pdf(LOCAL / "source.pdf", manifest, encoder)
    if not {str(i) for i in range(1, 25)}.issubset(
        {c.rule_number.split(".")[1] for c in chunks if c.rule_number.startswith("II.")}
    ):
        raise ValueError("CHAPTER_II_COVERAGE_GAP")
    documents = [f"{c.rule_number} {c.heading}. {c.text}" for c in chunks]
    embeddings = encoder.embed_documents(documents)
    store = RuleStore(
        os.environ.get(
            "CUEORBIT_RULES_DATABASE_URL",
            "postgresql+asyncpg://replay@127.0.0.1:54331/cueorbit_replay_m1_m2",
        )
    )
    try:
        await store.initialize()
        result = await store.import_corpus(manifest, chunks, embeddings)
        contexts = await store.list_contexts()
        active_context = await store.context_for(manifest)
    finally:
        await store.close()
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / "chunks.json").write_text(
        json.dumps(
            [c.model_dump(mode="json") for c in chunks], ensure_ascii=False, indent=2
        )
        + "\n"
    )
    metadata = {
        "source": manifest.model_dump(mode="json"),
        "contexts": [c.model_dump(mode="json") for c in contexts],
        "active_context": active_context.model_dump(mode="json"),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "rule_number": c.rule_number,
                "heading": c.heading,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text_sha256": c.text_sha256,
                "body_tokens": encoder.count(c.text),
                "input_tokens": encoder.count("passage: " + doc),
            }
            for c, doc in zip(chunks, documents, strict=True)
        ],
    }
    (ROOT / "data/rules/chunk-manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "chunk_count": len(chunks),
                "rule_count": len({c.rule_number for c in chunks}),
                "contexts": [c.model_dump(mode="json") for c in contexts],
                "max_input_tokens": max(
                    item["input_tokens"] for item in metadata["chunks"]
                ),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Explicit local rule corpus operations"
    )
    parser.add_argument(
        "operation",
        choices=("fetch-model", "fetch-source", "build-index", "verify-source"),
    )
    args = parser.parse_args()
    if args.operation == "fetch-model":
        from .artifacts import fetch_model

        fetch_model(MODEL_ROOT)
        print(json.dumps({"status": "verified", "artifact": "pinned-local-encoder"}))
    elif args.operation == "fetch-source":
        fetch_source(LOCAL / "source.pdf")
        MANIFEST_PATH.write_text(source_manifest().model_dump_json(indent=2) + "\n")
        print(json.dumps({"status": "verified", "source_sha256": PDF_SHA256}))
    elif args.operation == "verify-source":
        valid = (
            hashlib.sha256((LOCAL / "source.pdf").read_bytes()).hexdigest()
            == source_manifest().pdf_sha256
        )
        if not valid:
            raise ValueError("SOURCE_HASH_MISMATCH")
        print(json.dumps({"status": "verified", "source_sha256": PDF_SHA256}))
    else:
        asyncio.run(build_index())


if __name__ == "__main__":
    main()
