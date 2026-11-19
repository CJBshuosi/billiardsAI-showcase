"""Explicit fixed-source download; never called from a question request."""

import hashlib
from contextlib import nullcontext
from pathlib import Path

import httpx

from .contracts import PDF_SHA256, SOURCE_URL


def fetch_source(destination: Path, *, client: httpx.Client | None = None):
    manager = (
        nullcontext(client)
        if client is not None
        else httpx.Client(follow_redirects=False, trust_env=False, timeout=30)
    )
    with manager as connection:
        with connection.stream("GET", SOURCE_URL, follow_redirects=False) as response:
            response.raise_for_status()
            data = bytearray()
            for block in response.iter_bytes():
                data.extend(block)
                if len(data) > 10 * 1024 * 1024:
                    raise ValueError("SOURCE_TOO_LARGE")
    if not data.startswith(b"%PDF") or hashlib.sha256(data).hexdigest() != PDF_SHA256:
        raise ValueError("SOURCE_HASH_MISMATCH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != PDF_SHA256:
            raise ValueError("SOURCE_CACHE_CONFLICT")
        return
    temporary = destination.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(destination)
