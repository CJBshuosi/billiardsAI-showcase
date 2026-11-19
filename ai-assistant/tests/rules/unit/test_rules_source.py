import httpx
import pytest

from cueorbit_replay.rules.source import fetch_source


@pytest.mark.parametrize(
    "status,body,headers",
    [
        (302, b"", {"location": "https://elsewhere.invalid/source.pdf"}),
        (200, b"%PDF-changed", {"content-type": "application/pdf"}),
        (503, b"unavailable", {}),
    ],
)
def test_source_failure_never_replaces_cached_pdf(tmp_path, status, body, headers):
    destination = tmp_path / "source.pdf"
    destination.write_bytes(b"previous-cache")
    seen = []

    def handle(request):
        seen.append(str(request.url))
        return httpx.Response(status, content=body, headers=headers)

    with httpx.Client(
        transport=httpx.MockTransport(handle), follow_redirects=False
    ) as client:
        with pytest.raises((ValueError, httpx.HTTPError)):
            fetch_source(destination, client=client)
    assert destination.read_bytes() == b"previous-cache"
    assert seen == [
        "https://wpapool.com/wp-content/uploads/2025/08/250816-Rules-of-Heyball.pdf"
    ]
