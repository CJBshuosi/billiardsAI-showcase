import hashlib
import http.client
import socket
import ssl
import urllib.error
import urllib.request

import pytest

from billiards_ai.lab.dependency_audit import transport as transport_module
from billiards_ai.lab.dependency_audit.transport import (
    HttpResponse,
    MetadataResponse,
    MetadataSecurityError,
    PyPIMetadataClient,
    RejectRedirects,
    simple_url,
    version_url,
)


class StaticTransport:
    def __init__(self, response: HttpResponse):
        self.response = response

    def get(self, url: str, *, accept: str, bytes_max: int) -> HttpResponse:
        return self.response


def test_response_compatibility_alias_is_the_immutable_response_type() -> None:
    response = HttpResponse(
        requested_url="https://pypi.org/simple/demo/",
        final_url="https://pypi.org/simple/demo/",
        content_type="application/vnd.pypi.simple.v1+json",
        body=b"{}",
        redirect_count=0,
    )

    assert MetadataResponse is HttpResponse
    with pytest.raises(AttributeError):
        response.body = b"changed"  # type: ignore[misc]


def test_client_rejects_cross_host_final_url() -> None:
    client = PyPIMetadataClient(
        transport=StaticTransport(
            HttpResponse(
                requested_url="https://pypi.org/simple/demo/",
                final_url="https://files.pythonhosted.org/demo.whl",
                content_type="application/vnd.pypi.simple.v1+json",
                body=b"{}",
                redirect_count=1,
            )
        ),
        single_response_bytes_max=1024,
        total_metadata_bytes_max=2048,
    )

    with pytest.raises(MetadataSecurityError, match="redirect or final URL mismatch"):
        client.simple("demo")


def test_client_rejects_archive_path_even_on_pypi_host() -> None:
    with pytest.raises(MetadataSecurityError, match="metadata path is not allowed"):
        PyPIMetadataClient.validate_url("https://pypi.org/packages/demo.whl")


def test_client_rejects_url_components_outside_the_exact_allowlist() -> None:
    for url in (
        "http://pypi.org/simple/demo/",
        "https://user:password@pypi.org/simple/demo/",
        "https://pypi.org/simple/demo/?download=1",
        "https://pypi.org/simple/demo/#fragment",
        "https://pypi.org/simple/demo",
        "https://pypi.org/pypi/demo/1.2.0/json/extra",
    ):
        with pytest.raises(MetadataSecurityError):
            PyPIMetadataClient.validate_url(url)


def test_client_rejects_percent_encoded_path_separators() -> None:
    with pytest.raises(MetadataSecurityError, match="metadata path is not allowed"):
        PyPIMetadataClient.validate_url("https://pypi.org/simple/demo%2fextra/")


@pytest.mark.parametrize(
    ("single_response_bytes_max", "total_metadata_bytes_max"),
    [
        (12 * 1024 * 1024 + 1, 128 * 1024 * 1024),
        (12 * 1024 * 1024, 128 * 1024 * 1024 + 1),
    ],
)
def test_client_rejects_constructor_limits_above_the_mandated_hard_caps(
    single_response_bytes_max: int, total_metadata_bytes_max: int
) -> None:
    response = HttpResponse(
        requested_url="https://pypi.org/simple/demo/",
        final_url="https://pypi.org/simple/demo/",
        content_type="application/vnd.pypi.simple.v1+json",
        body=b"{}",
        redirect_count=0,
    )

    with pytest.raises(ValueError, match="hard caps"):
        PyPIMetadataClient(
            StaticTransport(response), single_response_bytes_max, total_metadata_bytes_max
        )


def test_client_allows_total_metadata_limit_above_64_mib() -> None:
    response = HttpResponse(
        requested_url="https://pypi.org/simple/demo/",
        final_url="https://pypi.org/simple/demo/",
        content_type="application/vnd.pypi.simple.v1+json",
        body=b"{}",
        redirect_count=0,
    )
    client = PyPIMetadataClient(
        StaticTransport(response),
        single_response_bytes_max=12 * 1024 * 1024,
        total_metadata_bytes_max=64 * 1024 * 1024 + 1,
    )

    assert client.simple("demo") == b"{}"


def test_client_enforces_cumulative_response_limit() -> None:
    response = HttpResponse(
        requested_url="https://pypi.org/simple/demo/",
        final_url="https://pypi.org/simple/demo/",
        content_type="application/vnd.pypi.simple.v1+json",
        body=b"123456",
        redirect_count=0,
    )
    client = PyPIMetadataClient(StaticTransport(response), 10, 10)

    client.simple("demo")

    with pytest.raises(MetadataSecurityError, match="cumulative metadata limit"):
        client.simple("demo")


def test_client_rejects_oversized_response_and_wrong_content_type() -> None:
    oversized = HttpResponse(
        requested_url="https://pypi.org/simple/demo/",
        final_url="https://pypi.org/simple/demo/",
        content_type="application/vnd.pypi.simple.v1+json",
        body=b"123456",
        redirect_count=0,
    )
    client = PyPIMetadataClient(StaticTransport(oversized), 5, 10)

    with pytest.raises(MetadataSecurityError, match="single metadata response limit"):
        client.simple("demo")

    wrong_type = HttpResponse(
        requested_url="https://pypi.org/simple/demo/",
        final_url="https://pypi.org/simple/demo/",
        content_type="text/html",
        body=b"{}",
        redirect_count=0,
    )
    client = PyPIMetadataClient(StaticTransport(wrong_type), 10, 10)

    with pytest.raises(MetadataSecurityError, match="unexpected metadata content type"):
        client.simple("demo")


def test_client_records_immutable_evidence_for_valid_metadata() -> None:
    response = HttpResponse(
        requested_url="https://pypi.org/simple/demo/",
        final_url="https://pypi.org/simple/demo/",
        content_type="application/vnd.pypi.simple.v1+json; charset=utf-8",
        body=b'{"meta":{"api-version":"1.0"}}',
        redirect_count=0,
    )
    client = PyPIMetadataClient(StaticTransport(response), 1024, 2048)

    assert client.simple("demo") == response.body
    assert client.metadata_accesses[0].url == "https://pypi.org/simple/demo/"
    assert client.metadata_accesses[0].content_type == response.content_type
    assert client.metadata_accesses[0].bytes == len(response.body)
    assert client.metadata_accesses[0].sha256 == hashlib.sha256(response.body).hexdigest()
    with pytest.raises(AttributeError):
        client.metadata_accesses += ()  # type: ignore[misc]


def test_metadata_url_constructors_normalize_names_and_versions() -> None:
    assert simple_url("Demo_Package") == "https://pypi.org/simple/demo-package/"
    assert version_url("Demo_Package", "1.02.0") == "https://pypi.org/pypi/demo-package/1.2.0/json"


def test_redirect_handler_rejects_a_redirect_without_accessing_network() -> None:
    with pytest.raises(
        MetadataSecurityError, match="redirect rejected: 302 https://elsewhere.invalid/"
    ):
        RejectRedirects().redirect_request(
            None, None, 302, "Found", None, "https://elsewhere.invalid/"
        )


class _OfflineUrllibResponse:
    def __init__(self, url: str) -> None:
        self._url = url
        self.headers = {"Content-Type": "application/vnd.pypi.simple.v1+json"}

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return b"{}"

    def geturl(self) -> str:
        return self._url


def test_final_fix_09_urllib_transport_uses_a_finite_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://pypi.org/simple/demo/"

    class OfflineOpener:
        def __init__(self) -> None:
            self.timeout: float | None = None

        def open(self, request: urllib.request.Request, *, timeout: float):
            self.timeout = timeout
            return _OfflineUrllibResponse(request.full_url)

    opener = OfflineOpener()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_handlers: opener)
    transport = transport_module.UrllibMetadataTransport(timeout_seconds=7.5)

    response = transport.get(
        url,
        accept="application/vnd.pypi.simple.v1+json",
        bytes_max=1024,
    )

    assert response.body == b"{}"
    assert opener.timeout == 7.5
    assert 0 < opener.timeout < 60


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(socket.gaierror("DNS failed")),
        urllib.error.HTTPError("https://pypi.org/simple/demo/", 503, "unavailable", {}, None),
        TimeoutError("timed out"),
        ssl.SSLError("TLS failed"),
        ConnectionError("connection failed"),
        http.client.IncompleteRead(b"partial", 10),
    ],
)
def test_final_fix_09_expected_live_transport_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    class RejectingOpener:
        def open(self, _request: urllib.request.Request, *, timeout: float):
            raise error

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: RejectingOpener(),
    )
    typed_error = transport_module.MetadataTransportError

    with pytest.raises(typed_error, match="metadata transport failed"):
        transport_module.UrllibMetadataTransport(timeout_seconds=5).get(
            "https://pypi.org/simple/demo/",
            accept="application/vnd.pypi.simple.v1+json",
            bytes_max=1024,
        )


def test_final_fix_09_unexpected_urllib_os_error_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenOpener:
        def open(self, _request: urllib.request.Request, *, timeout: float):
            raise OSError("programming defect")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_handlers: BrokenOpener())

    with pytest.raises(OSError, match="programming defect"):
        transport_module.UrllibMetadataTransport(timeout_seconds=5).get(
            "https://pypi.org/simple/demo/",
            accept="application/vnd.pypi.simple.v1+json",
            bytes_max=1024,
        )
