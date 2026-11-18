import importlib.util
from pathlib import Path

from scripts.development import blackbox_validation as validation

ENTRY = Path(__file__).resolve().parents[3] / "scripts/development/blackbox_media_diagnostic.py"


def test_codec_path_is_valid_in_initial_child_environment_without_writing_old_run(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location("diag_env_test", ENTRY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    run = tmp_path / "run"
    (run / "codec").mkdir(parents=True)
    dll = run / "codec/openh264-2.5.0-win64.dll"
    dll.write_bytes(b"untouched fixture")
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "new-output"
    output.mkdir()
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "system"))
    env = m.diagnostic_environment(run, output, {}, validation)
    assert env["OPENH264_LIBRARY"] == str(dll)
    assert Path(env["OPENH264_LIBRARY"]).is_file()
    assert all(
        Path(env[k]).is_relative_to(output)
        for k in ("TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
    )
    assert {
        str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()
    } == before
