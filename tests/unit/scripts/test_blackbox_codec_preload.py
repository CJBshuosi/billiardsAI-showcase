import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[3] / "scripts/development/blackbox_codec_diagnostic.py"


def module():
    spec = importlib.util.spec_from_file_location("codec_preload_test", ENTRY)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_preload_uses_verified_absolute_path_and_retains_library(tmp_path, monkeypatch):
    m = module()
    raw = b"controlled fake native library"
    dll = tmp_path / "codec.dll"
    dll.write_bytes(raw)
    monkeypatch.setattr(
        m,
        "CODEC_PIN",
        {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
    )
    calls = []
    handle = object()
    rows = []
    profile = m.helper.Profiler(rows.append)

    def loader(path):
        calls.append(path)
        return handle

    assert m.preload_codec(dll, profile, loader=loader) is handle
    assert calls == [str(dll)]
    assert rows[-1]["stage"] == "CODEC_PRELOAD" and rows[-1]["opened"] is True
    dll.write_bytes(b"X" * len(raw))
    with pytest.raises(ValueError, match="CODEC_IDENTITY"):
        m.preload_codec(dll, profile, loader=loader)
    assert calls == [str(dll)]


def test_preload_failure_exports_only_error_number(tmp_path, monkeypatch):
    m = module()
    raw = b"controlled fake native library"
    dll = tmp_path / "codec.dll"
    dll.write_bytes(raw)
    monkeypatch.setattr(
        m,
        "CODEC_PIN",
        {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
    )
    rows = []
    profile = m.helper.Profiler(rows.append)

    def loader(path):
        error = OSError("private dependency path")
        error.winerror = 126
        raise error

    assert m.preload_codec(dll, profile, loader=loader) is None
    exported = m.export_events("\n".join(json.dumps(row) for row in rows).encode())
    assert exported[-1]["winerror"] == 126 and exported[-1]["opened"] is False
    assert "private" not in str(exported)
