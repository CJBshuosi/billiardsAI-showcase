"""Metadata-only PyPI transport with a closed URL allowlist."""

import http.client
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
from urllib.parse import urlsplit

from packaging.utils import canonicalize_name
from packaging.version import Version

from billiards_ai.lab.dependency_audit.models import MetadataAccess


class MetadataSecurityError(ValueError):
    """Raised when a metadata request violates the closed transport policy."""


class MetadataTransportError(RuntimeError):
    """Raised for expected failures at the live urllib metadata boundary."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    requested_url: str
    final_url: str
    content_type: str
    body: bytes
    redirect_count: int


MetadataResponse = HttpResponse


class MetadataTransport(Protocol):
    def get(self, url: str, *, accept: str, bytes_max: int) -> HttpResponse: ...


def simple_url(name: str) -> str:
    return f"https://pypi.org/simple/{canonicalize_name(name)}/"


def version_url(name: str, version: str) -> str:
    normalized = canonicalize_name(name)
    parsed_version = Version(version)
    return f"https://pypi.org/pypi/{normalized}/{parsed_version}/json"


def _is_normalized_name(value: str) -> bool:
    return bool(value) and value == canonicalize_name(value)


def _is_allowed_metadata_path(path: str) -> bool:
    segments = path.split("/")
    if len(segments) >= 4 and segments[:2] == ["", "simple"] and segments[-1] == "":
        return len(segments) == 4 and _is_normalized_name(segments[2])
    if len(segments) != 5 or segments[:2] != ["", "pypi"] or segments[-1] != "json":
        return False
    name, version = segments[2], segments[3]
    if not _is_normalized_name(name):
        return False
    try:
        return str(Version(version)) == version
    except ValueError:
        return False


def _validate_metadata_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise MetadataSecurityError("metadata URL must use https")
    if parsed.netloc != "pypi.org" or parsed.hostname != "pypi.org":
        raise MetadataSecurityError("metadata host is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise MetadataSecurityError("metadata credentials are not allowed")
    if parsed.query or parsed.fragment:
        raise MetadataSecurityError("metadata query strings and fragments are not allowed")
    if "%" in parsed.path:
        raise MetadataSecurityError("metadata path is not allowed")
    if not _is_allowed_metadata_path(parsed.path):
        raise MetadataSecurityError("metadata path is not allowed")


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise MetadataSecurityError(f"redirect rejected: {code} {newurl}")


class UrllibMetadataTransport:
    """Standard-library transport that never follows redirects."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        if not isinstance(timeout_seconds, int | float) or not 0 < timeout_seconds < 60:
            raise ValueError("metadata timeout must be finite and between 0 and 60 seconds")
        self._timeout_seconds = float(timeout_seconds)

    def get(self, url: str, *, accept: str, bytes_max: int) -> HttpResponse:
        _validate_metadata_url(url)
        request = urllib.request.Request(url, headers={"Accept": accept})
        opener = urllib.request.build_opener(RejectRedirects())
        try:
            with opener.open(request, timeout=self._timeout_seconds) as response:
                body = response.read(bytes_max + 1)
                content_type = response.headers.get("Content-Type", "")
                return HttpResponse(
                    requested_url=url,
                    final_url=response.geturl(),
                    content_type=content_type,
                    body=body,
                    redirect_count=0,
                )
        except MetadataSecurityError:
            raise
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ssl.SSLError,
            ConnectionError,
            http.client.HTTPException,
        ) as error:
            raise MetadataTransportError(
                f"metadata transport failed: {type(error).__name__}: {error}"
            ) from error


class PyPIMetadataClient:
    """Closed metadata client for the PyPI simple and version JSON endpoints."""

    _SIMPLE_CONTENT_TYPE = "application/vnd.pypi.simple.v1+json"
    _VERSION_CONTENT_TYPE = "application/json"
    _MAX_SINGLE_METADATA_BYTES = 12 * 1024 * 1024
    _MAX_TOTAL_METADATA_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        transport: MetadataTransport,
        single_response_bytes_max: int,
        total_metadata_bytes_max: int,
    ) -> None:
        if single_response_bytes_max <= 0 or total_metadata_bytes_max <= 0:
            raise ValueError("metadata byte limits must be positive")
        if (
            single_response_bytes_max > self._MAX_SINGLE_METADATA_BYTES
            or total_metadata_bytes_max > self._MAX_TOTAL_METADATA_BYTES
        ):
            raise ValueError("metadata byte limits exceed mandated hard caps")
        self._transport = transport
        self._single_response_bytes_max = single_response_bytes_max
        self._total_metadata_bytes_max = total_metadata_bytes_max
        self._total_metadata_bytes = 0
        self._metadata_accesses: list[MetadataAccess] = []

    @property
    def metadata_accesses(self) -> tuple[MetadataAccess, ...]:
        return tuple(self._metadata_accesses)

    @staticmethod
    def validate_url(url: str) -> None:
        _validate_metadata_url(url)

    def simple(self, name: str) -> bytes:
        return self._fetch(simple_url(name), self._SIMPLE_CONTENT_TYPE)

    def version(self, name: str, version: str) -> bytes:
        return self._fetch(version_url(name, version), self._VERSION_CONTENT_TYPE)

    def _fetch(self, url: str, expected_content_type: str) -> bytes:
        self.validate_url(url)
        response = self._transport.get(
            url,
            accept=expected_content_type,
            bytes_max=self._single_response_bytes_max,
        )
        if (
            response.requested_url != url
            or response.final_url != url
            or response.redirect_count != 0
        ):
            raise MetadataSecurityError("redirect or final URL mismatch")
        response_type = response.content_type.split(";", maxsplit=1)[0].strip().lower()
        if response_type != expected_content_type:
            raise MetadataSecurityError("unexpected metadata content type")
        response_bytes = len(response.body)
        if response_bytes > self._single_response_bytes_max:
            raise MetadataSecurityError("single metadata response limit exceeded")
        if self._total_metadata_bytes + response_bytes > self._total_metadata_bytes_max:
            raise MetadataSecurityError("cumulative metadata limit exceeded")
        self._total_metadata_bytes += response_bytes
        self._metadata_accesses.append(
            MetadataAccess(
                url=url,
                content_type=response.content_type,
                bytes=response_bytes,
                sha256=sha256(response.body).hexdigest(),
            )
        )
        return response.body
