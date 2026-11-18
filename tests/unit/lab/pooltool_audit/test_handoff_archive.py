import hashlib
import json
import zipfile
from pathlib import Path

from billiards_ai.lab.pooltool_audit.handoff import DownloadedWheel
from billiards_ai.lab.pooltool_audit.handoff_archive import append_handoff_payloads


def test_appends_verified_wheels_packet_and_fixed_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "handoff.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("billiardsAI-mainline1/README.md", b"source")
    wheel_path = tmp_path / "fixture.whl"
    wheel_path.write_bytes(b"wheel")
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(b'{"packet":"fixture"}\n')
    wheel = DownloadedWheel(
        filename="fixture.whl",
        path=wheel_path,
        bytes=5,
        sha256=hashlib.sha256(b"wheel").hexdigest(),
    )

    manifest = append_handoff_payloads(
        archive,
        packet_path=packet_path,
        packet_sha256="a" * 64,
        source_commit="b" * 40,
        wheels=(wheel,),
    )

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names))
        assert bundle.read("billiardsAI-mainline1/_handoff/wheelhouse/fixture.whl") == b"wheel"
        stored = json.loads(bundle.read("billiardsAI-mainline1/_handoff/HANDOFF.json"))
    assert stored == manifest
    assert stored["payload_count"] == 1
    assert stored["payload_bytes"] == 5
    assert stored["network_required_on_raytheon"] is False


def test_bundle_metadata_is_reproducible(tmp_path: Path) -> None:
    wheel_path = tmp_path / "fixture.whl"
    wheel_path.write_bytes(b"wheel")
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(b"{}\n")
    wheel = DownloadedWheel(
        filename="fixture.whl",
        path=wheel_path,
        bytes=5,
        sha256=hashlib.sha256(b"wheel").hexdigest(),
    )
    archives = []
    for name in ("one.zip", "two.zip"):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as output:
            info = zipfile.ZipInfo("billiardsAI-mainline1/README.md", (1980, 1, 1, 0, 0, 0))
            output.writestr(info, b"source")
        append_handoff_payloads(
            path,
            packet_path=packet_path,
            packet_sha256="a" * 64,
            source_commit="b" * 40,
            wheels=(wheel,),
        )
        archives.append(path.read_bytes())
    assert archives[0] == archives[1]
