from pathlib import Path

SCRIPT = Path("scripts/algorithm/pooltool_hash_probe.py")


def test_pooltool_hash_probe_exposes_only_hash_and_packet_arguments() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--draft",' in text
    assert 'parser.add_argument("--draft-sha256",' in text
    assert 'parser.add_argument("--cache-root",' in text
    assert 'parser.add_argument("--output-root",' in text
    assert "pip install" not in text
    assert "import pooltool" not in text
    assert "download_panda_hash" in text
    assert "publish_pooltool_packet" in text
