"""Build and publish a metadata-only pooltool payload draft."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit.transport import PyPIMetadataClient
from billiards_ai.lab.pooltool_audit.archive import (
    PandaArchiveHead,
    PandaArchiveLink,
    discover_panda3d_wheel,
    select_panda3d_wheel,
)
from billiards_ai.lab.pooltool_audit.models import (
    PooltoolAuditRequest,
    PooltoolMetadataDraft,
    archive_access,
)
from billiards_ai.lab.pooltool_audit.resolver import resolve_pooltool_pypi_closure


class PooltoolAuditError(RuntimeError):
    """Raised when the metadata draft cannot be safely built or published."""


class PandaMetadataClient(Protocol):
    def index(self) -> tuple[bytes, str]: ...

    def head(self, link: PandaArchiveLink) -> PandaArchiveHead: ...


def load_request(path: Path) -> PooltoolAuditRequest:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return PooltoolAuditRequest.model_validate(payload)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise PooltoolAuditError(f"pooltool audit request is invalid: {path}") from error


def build_metadata_draft(
    request: PooltoolAuditRequest,
    *,
    pypi_client: PyPIMetadataClient,
    panda_client: PandaMetadataClient,
) -> PooltoolMetadataDraft:
    closure = resolve_pooltool_pypi_closure(request, pypi_client)
    index_payload, content_type = panda_client.index()
    link = discover_panda3d_wheel(
        index_payload,
        target=request.target,
        version="1.11.0.dev3702",
    )
    panda = select_panda3d_wheel(
        index_payload,
        target=request.target,
        version="1.11.0.dev3702",
        head=panda_client.head(link),
    )
    pypi_metadata_bytes = sum(access.bytes for access in pypi_client.metadata_accesses)
    return PooltoolMetadataDraft(
        request=request,
        closure=closure,
        panda3d=panda,
        pypi_metadata_bytes=pypi_metadata_bytes,
        panda_index_access=archive_access(index_payload, content_type),
        payload_count=len(closure.packages) + 1,
        payload_bytes=sum(package.wheel.bytes for package in closure.packages) + panda.bytes,
        approval_status="panda3d_sha256_required",
        side_effects={
            "archive_downloads": 0,
            "installs": 0,
            "package_imports": 0,
            "weights_loaded": 0,
        },
    )


def canonical_draft_bytes(draft: PooltoolMetadataDraft) -> bytes:
    return (
        json.dumps(
            draft.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def publish_draft(draft: PooltoolMetadataDraft, output_root: Path) -> Path:
    payload = canonical_draft_bytes(draft)
    draft_id = sha256(payload).hexdigest()
    destination = output_root.resolve() / draft.request.request_id / draft_id
    if destination.exists():
        existing = destination / "draft.json"
        if existing.is_file() and existing.read_bytes() == payload:
            return destination
        raise PooltoolAuditError("pooltool audit draft destination conflicts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise PooltoolAuditError("pooltool audit temporary destination already exists")
    temporary.mkdir()
    try:
        (temporary / "draft.json").write_bytes(payload)
        temporary.rename(destination)
    except OSError as error:
        raise PooltoolAuditError("pooltool audit draft publication failed") from error
    return destination
