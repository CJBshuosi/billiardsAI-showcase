"""Closed live metadata transport for the official Panda3D archive."""

from __future__ import annotations

import http.client
import ssl
import urllib.error
import urllib.request

from billiards_ai.lab.pooltool_audit.archive import PandaArchiveHead, PandaArchiveLink

PANDA_INDEX_URL = "https://archive.panda3d.org/simple/panda3d/"


class PandaMetadataTransportError(RuntimeError):
    """Raised when the official archive metadata boundary fails closed."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise PandaMetadataTransportError(f"Panda3D metadata redirect rejected: {code}")


class PandaArchiveMetadataClient:
    def __init__(self, *, timeout_seconds: float = 30, index_bytes_max: int = 2_000_000) -> None:
        if not 0 < timeout_seconds < 60 or not 0 < index_bytes_max <= 2_000_000:
            raise ValueError("Panda3D metadata transport limits are invalid")
        self._timeout = timeout_seconds
        self._index_bytes_max = index_bytes_max
        self._opener = urllib.request.build_opener(_RejectRedirects())

    def index(self) -> tuple[bytes, str]:
        request = urllib.request.Request(
            PANDA_INDEX_URL,
            headers={"Accept": "text/html"},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                if response.geturl() != PANDA_INDEX_URL:
                    raise PandaMetadataTransportError("Panda3D index final URL mismatch")
                payload = response.read(self._index_bytes_max + 1)
                content_type = response.headers.get("Content-Type", "")
        except PandaMetadataTransportError:
            raise
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ssl.SSLError,
            ConnectionError,
            http.client.HTTPException,
        ) as error:
            raise PandaMetadataTransportError(
                f"Panda3D index request failed: {type(error).__name__}"
            ) from error
        if len(payload) > self._index_bytes_max:
            raise PandaMetadataTransportError("Panda3D index metadata limit exceeded")
        if content_type.split(";", 1)[0].strip().lower() != "text/html":
            raise PandaMetadataTransportError("Panda3D index content type is invalid")
        return payload, content_type

    def head(self, link: PandaArchiveLink) -> PandaArchiveHead:
        request = urllib.request.Request(link.url, method="HEAD")
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                if response.geturl() != link.url:
                    raise PandaMetadataTransportError("Panda3D wheel HEAD final URL mismatch")
                raw_length = response.headers.get("Content-Length")
                last_modified = response.headers.get("Last-Modified", "")
                etag = response.headers.get("ETag", "")
        except PandaMetadataTransportError:
            raise
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ssl.SSLError,
            ConnectionError,
            http.client.HTTPException,
        ) as error:
            raise PandaMetadataTransportError(
                f"Panda3D wheel HEAD failed: {type(error).__name__}"
            ) from error
        try:
            size = int(raw_length) if raw_length is not None else 0
        except ValueError as error:
            raise PandaMetadataTransportError("Panda3D wheel size is invalid") from error
        if size <= 0 or not last_modified or not etag:
            raise PandaMetadataTransportError("Panda3D wheel HEAD metadata is incomplete")
        return PandaArchiveHead(
            url=link.url,
            bytes=size,
            last_modified=last_modified,
            etag=etag,
        )
