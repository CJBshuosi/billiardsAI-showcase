"""Hash-bound editorial warnings, independent of official quotes and vectors."""

import hashlib
from functools import lru_cache

from ..fixtures import ROOT
from .contracts import AnnotationRegistry, SourceAnnotation, SourceAnnotationNotice

__all__ = ["AnnotationRegistry", "SourceAnnotation", "notices_for_citations"]


@lru_cache(maxsize=1)
def load_annotations():
    return AnnotationRegistry.model_validate_json(
        (ROOT / "data/rules/source-annotations.json").read_text()
    )


def _bound(chunk, annotation):
    if (
        chunk.source_id != annotation.source_id
        or chunk.text_sha256 != annotation.text_sha256
        or hashlib.sha256(chunk.text.encode()).hexdigest() != annotation.text_sha256
        or chunk.text.count(annotation.fragment) != 1
    ):
        raise ValueError("ANNOTATION_SOURCE_MISMATCH")
    return chunk.text.index(annotation.fragment)


def annotation_notices(chunks, *, registry=None):
    if not chunks:
        return ()
    registry = registry or load_annotations()
    notices = []
    for chunk in chunks:
        for ann in registry.annotations:
            if ann.chunk_id != chunk.chunk_id:
                continue
            _bound(chunk, ann)
            notices.append(
                SourceAnnotationNotice(
                    annotation_id=ann.annotation_id,
                    registry_version=registry.version,
                    chunk_id=chunk.chunk_id,
                    rule_number=chunk.rule_number,
                    version=chunk.version,
                    fragment=ann.fragment,
                    kind=ann.kind,
                    note=ann.note,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_url=chunk.source_url,
                )
            )
    return tuple(notices)


def notices_for_citations(citations, chunks, *, registry=None):
    if not citations or not chunks:
        return ()
    registry = registry or load_annotations()
    notices = annotation_notices(chunks, registry=registry)
    pool = {c.chunk_id: c for c in chunks}
    selected = {}
    for draft in citations:
        identity = draft["chunk_id"] if isinstance(draft, dict) else draft.chunk_id
        quote = draft["quote"] if isinstance(draft, dict) else draft.quote
        chunk = pool.get(identity)
        if chunk is None:
            continue
        for notice in notices:
            if notice.chunk_id != identity:
                continue
            left = chunk.text.index(notice.fragment)
            right = left + len(notice.fragment)
            # Every occurrence counts; ambiguous duplicate short quotes fail closed.
            start = 0
            while (offset := chunk.text.find(quote, start)) >= 0:
                if offset < right and offset + len(quote) > left:
                    selected[notice.annotation_id] = notice
                start = offset + 1
    return tuple(selected.values())


def validate_registry_metadata(registry, metadata):
    if registry.corpus_hash != metadata["active_context"]["corpus_hash"]:
        raise ValueError("ANNOTATION_CORPUS_MISMATCH")
    chunks = {c["chunk_id"]: c for c in metadata["chunks"]}
    source = metadata["source"]
    for annotation in registry.annotations:
        chunk = chunks.get(annotation.chunk_id)
        if (
            chunk is None
            or chunk["text_sha256"] != annotation.text_sha256
            or source["source_id"] != annotation.source_id
            or source["pdf_sha256"] != annotation.pdf_sha256
        ):
            raise ValueError("ANNOTATION_METADATA_MISMATCH")
