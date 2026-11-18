import hashlib
import json

import pytest

from scripts.algorithm import recording_packet as packet


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "/x",
        "a\\b",
        "C:x",
        "aux.txt",
        "a/NUL.py",
        "a./x",
        "a /x",
        ".PROJECT-ARCHIVE/x",
        ".git/config",
        "a?b",
        "a|b",
        'a"b',
        "a\nb",
        "a\x7fb",
    ],
)
def test_windows_packet_rejects_aliases_and_protected_names(name):
    with pytest.raises(ValueError):
        packet.safe_member(name)


def _packet(tmp_path):
    root = tmp_path / "packet"
    root.mkdir()
    content = b"recording source\n"
    (root / "source.py").write_bytes(content)
    manifest = dict(
        contract="recording-first-windows-packet/v1",
        members={"source.py": dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())},
    )
    raw = json.dumps(manifest).encode()
    (root / "packet-manifest.json").write_bytes(raw)
    return root, hashlib.sha256(raw).hexdigest()


def test_packet_verification_rejects_changed_or_unlisted_content(tmp_path):
    root, digest = _packet(tmp_path)
    assert packet.verify_packet(root, digest)["contract"] == "recording-first-windows-packet/v1"
    (root / "extra.py").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory"):
        packet.verify_packet(root, digest)
    (root / "extra.py").unlink()
    (root / "source.py").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="content"):
        packet.verify_packet(root, digest)


def test_packet_refuses_reparse_and_case_aliases_before_reading_payload(tmp_path):
    root, digest = _packet(tmp_path)
    (root / "unlisted").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        packet.verify_packet(root, digest)


def test_manifest_digest_is_checked_before_trusting_inventory(tmp_path):
    root, _ = _packet(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        packet.verify_packet(root, "f" * 64)


def test_case_alias_in_manifest_is_not_admitted(tmp_path):
    root, _ = _packet(tmp_path)
    manifest = json.loads((root / "packet-manifest.json").read_bytes())
    manifest["members"]["SOURCE.py"] = manifest["members"]["source.py"]
    raw = json.dumps(manifest).encode()
    (root / "packet-manifest.json").write_bytes(raw)
    with pytest.raises(ValueError, match="alias"):
        packet.verify_packet(root, hashlib.sha256(raw).hexdigest())
