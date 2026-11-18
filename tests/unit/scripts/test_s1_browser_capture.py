"""CLI transport contracts only; these fixtures never prove a browser recorded."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def tool():
    assert Path("scripts/development/s1_browser_capture.py").is_file()
    return importlib.import_module("scripts.development.s1_browser_capture")


def capture(tmp_path, runner):
    paths = []
    for name in ("agent-browser", "chrome", "ffmpeg"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths.append(path)
    return tool().BrowserCapture(*paths, tmp_path / "capture", runner=runner)


def test_capture_uses_an_owned_session_profile_and_argument_array(tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"success": True, "data": {}}), stderr=""
        )

    browser = capture(tmp_path, run)
    browser.start("http://127.0.0.1:8213/screen")
    assert len(calls) == 3
    assert calls[0][0][-2:] == ["open", "http://127.0.0.1:8213/screen"]
    assert calls[1][0][-5:] == ["set", "viewport", "1920", "1080", "1"]
    assert calls[2][0][-5:] == ["record", "start", str(browser.video), "--fps", "15"]
    config = json.loads((browser.root / "browser-config.json").read_bytes())
    assert config["profile"] == str(browser.root / "profile")
    assert config["headed"] is False and config["plugins"] == [] and config["extensions"] == []
    assert all("--session" in args and "--namespace" in args for args, _ in calls)
    assert all(not kw.get("shell", False) for _, kw in calls)
    assert all("AGENT_BROWSER_CDP" not in kw["env"] for _, kw in calls)


def test_native_utf8_browser_reply_and_logs_ignore_a_gbk_host_default(tmp_path, monkeypatch):
    payload = json.dumps(
        {"success": True, "data": {"title": "台球大屏 ✓"}}, ensure_ascii=False
    ).encode("utf-8")
    # Exercise actual subprocess decoding under the same default that failed
    # on BOYU; this is a transport fixture, not a browser/product success event.
    monkeypatch.setattr(subprocess, "_text_encoding", lambda: "gbk")

    def run(argv, **kwargs):
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import sys; sys.stdout.buffer.write(" + repr(payload) + ")",
            ],
            **kwargs,
        )

    browser = capture(tmp_path, run)
    assert browser.command("snapshot", "-i") == {"title": "台球大屏 ✓"}
    assert (browser.root / "command-0001.stdout.txt").read_bytes() == payload
    assert (browser.root / "command-0001.stderr.txt").read_bytes() == b""


def test_non_text_subprocess_reply_is_explicit_transport_failure(tmp_path):
    browser = capture(
        tmp_path, lambda *a, **k: SimpleNamespace(returncode=0, stdout=None, stderr="")
    )
    with pytest.raises(RuntimeError, match="UTF-8 text"):
        browser.command("snapshot", "-i")


@pytest.mark.parametrize("failure", ["exit", "malformed", "false", "missing"])
def test_transport_failure_never_becomes_capture_success(tmp_path, failure):
    def run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1 if failure == "exit" else 0,
            stdout="invalid"
            if failure == "malformed"
            else json.dumps({"success": False, "data": {}} if failure == "false" else {"data": {}}),
            stderr="controlled",
        )

    browser = capture(tmp_path, run)
    with pytest.raises(RuntimeError):
        browser.start("http://127.0.0.1:8213/screen")
    assert not browser.recording


def test_arbitrary_remote_target_is_rejected_before_browser_launch(tmp_path):
    calls = []
    browser = capture(tmp_path, lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="loopback"):
        browser.start("https://example.com/screen")
    assert calls == []


def test_close_attempts_browser_cleanup_even_when_record_stop_fails(tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[-2:] == ["record", "stop"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="controlled stop failure")
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"success": True, "data": {}}), stderr=""
        )

    browser = capture(tmp_path, run)
    browser.start("http://127.0.0.1:8213/screen")
    with pytest.raises(RuntimeError, match="stop"):
        browser.close()
    assert calls[-1][-1] == "close"
    assert not browser.closed, "a failed capture cannot claim successful finalization"


def test_existing_global_page_injection_config_is_rejected_without_launch(tmp_path, monkeypatch):
    module = tool()
    config = tmp_path / "owned-home" / ".agent-browser" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"initScripts": ["private unrelated script"]}))
    monkeypatch.setattr(module.Path, "home", classmethod(lambda cls: config.parent.parent))
    with pytest.raises(ValueError, match="global browser launch"):
        capture(tmp_path, lambda *a, **k: pytest.fail("must not launch a configured browser"))
