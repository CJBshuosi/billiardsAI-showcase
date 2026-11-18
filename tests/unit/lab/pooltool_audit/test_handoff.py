import hashlib
import json
from pathlib import Path

import pytest

from billiards_ai.lab.pooltool_audit.handoff import (
    PooltoolHandoffError,
    download_pooltool_wheels,
    load_pooltool_packet,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _packet(tmp_path: Path) -> tuple[Path, str, dict[str, bytes]]:
    pypi = b"pypi-wheel"
    panda = b"panda-wheel"
    payloads = {
        "https://files.pythonhosted.org/packages/aa/fixture-1.0-py3-none-any.whl": pypi,
        "https://buildbot.panda3d.org/downloads/"
        f"{'a' * 40}/panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl": panda,
    }
    document = {
        "schema_version": "1.0",
        "packet_sha256": "0" * 64,
        "draft_sha256": "1" * 64,
        "approval_status": "hash_complete",
        "payload_count": 2,
        "payload_bytes": len(pypi) + len(panda),
        "side_effects": {
            "archive_downloads": 1,
            "installs": 0,
            "package_imports": 0,
            "weights_loaded": 0,
        },
        "closure": {
            "packages": [
                {
                    "name": "fixture",
                    "version": "1.0",
                    "wheel": {
                        "filename": "fixture-1.0-py3-none-any.whl",
                        "url": next(iter(payloads)),
                        "bytes": len(pypi),
                        "sha256": hashlib.sha256(pypi).hexdigest(),
                    },
                }
            ]
        },
        "panda3d": {
            "filename": "panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl",
            "url": tuple(payloads)[1],
            "bytes": len(panda),
            "md5": hashlib.md5(panda, usedforsecurity=False).hexdigest(),
            "sha256": hashlib.sha256(panda).hexdigest(),
            "approval_status": "hash_complete",
        },
    }
    unhashed = {key: value for key, value in document.items() if key != "packet_sha256"}
    document["packet_sha256"] = hashlib.sha256(_canonical(unhashed)).hexdigest()
    path = tmp_path / "packet.json"
    path.write_bytes(_canonical(document))
    return path, document["packet_sha256"], payloads


def test_downloads_exact_packet_once_and_reuses_verified_cache(tmp_path: Path) -> None:
    path, identity, payloads = _packet(tmp_path)
    packet = load_pooltool_packet(path, identity)
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return payloads[url]

    first = download_pooltool_wheels(packet, tmp_path / "cache", fetch=fetch)
    second = download_pooltool_wheels(packet, tmp_path / "cache", fetch=fetch)

    assert len(first) == len(second) == 2
    assert len(calls) == 2
    assert sum(item.bytes for item in first) == packet["payload_bytes"]


def test_packet_tamper_and_payload_mismatch_fail_closed(tmp_path: Path) -> None:
    path, identity, payloads = _packet(tmp_path)
    document = json.loads(path.read_bytes())
    document["payload_bytes"] += 1
    path.write_bytes(_canonical(document))
    with pytest.raises(PooltoolHandoffError, match="packet SHA-256"):
        load_pooltool_packet(path, identity)

    path, identity, _ = _packet(tmp_path)
    packet = load_pooltool_packet(path, identity)
    with pytest.raises(PooltoolHandoffError, match="mismatch"):
        download_pooltool_wheels(packet, tmp_path / "bad", fetch=lambda _url: b"bad")
    assert not tuple((tmp_path / "bad").glob("*.whl"))


def test_rejects_unapproved_payload_host(tmp_path: Path) -> None:
    path, identity, _ = _packet(tmp_path)
    document = json.loads(path.read_bytes())
    document["closure"]["packages"][0]["wheel"]["url"] = "https://example.com/pkg.whl"
    unhashed = {key: value for key, value in document.items() if key != "packet_sha256"}
    document["packet_sha256"] = hashlib.sha256(_canonical(unhashed)).hexdigest()
    path.write_bytes(_canonical(document))

    with pytest.raises(PooltoolHandoffError, match="host"):
        load_pooltool_packet(path, document["packet_sha256"])
