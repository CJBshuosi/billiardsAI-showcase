"""Thin owned-session transport for actual page recording, never page rendering."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from scripts.development.blackbox_report import atomic_json
from scripts.development.blackbox_source import pin, plain

SCREEN_FPS = 15


class BrowserCapture:
    def __init__(self, cli, browser, ffmpeg, root, *, runner=subprocess.run):
        global_config = Path.home() / ".agent-browser" / "config.json"
        if global_config.is_file():
            with global_config.open("rb") as stream:
                raw = stream.read(128 * 1024 + 1)
            if len(raw) > 128 * 1024:
                raise ValueError("global browser launch configuration exceeds budget")
            settings = json.loads(raw)
            if not isinstance(settings, dict) or any(
                settings.get(key)
                for key in (
                    "extensions",
                    "plugins",
                    "initScripts",
                    "state",
                    "profile",
                    "provider",
                    "restore",
                    "cdp",
                    "autoConnect",
                )
            ):
                raise ValueError("global browser launch mutations cannot enter acceptance capture")
        self.cli, self.browser, self.ffmpeg = (plain(p) for p in (cli, browser, ffmpeg))
        self.root = plain(root)
        self.root.mkdir()
        self.identities = {name: pin(getattr(self, name)) for name in ("cli", "browser", "ffmpeg")}
        self.session = "s1capture-" + uuid4().hex
        self.runner = runner
        self.recording = self.started = self.closed = False
        self.sequence = 0
        self.video = self.root / "screen.mp4"
        self.config = self.root / "browser-config.json"
        atomic_json(
            self.config,
            {
                "headed": False,
                "executablePath": str(self.browser),
                "profile": str(self.root / "profile"),
                "plugins": [],
                "extensions": [],
                "autoConnect": False,
            },
        )
        self.environment = {
            k: v for k, v in os.environ.items() if not k.startswith("AGENT_BROWSER_")
        }
        self.environment["PATH"] = str(self.ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")

    def command(self, *arguments, timeout=30):
        self.sequence += 1
        argv = [
            str(self.cli),
            "--namespace",
            "s1-evidence",
            "--session",
            self.session,
            "--config",
            str(self.config),
            "--json",
            *map(str, arguments),
        ]
        result = self.runner(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout,
            env=self.environment,
        )
        if not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
            raise RuntimeError("browser command did not return UTF-8 text")
        # Local diagnostics stay separate from finite public acceptance summaries.
        (self.root / f"command-{self.sequence:04d}.stdout.txt").write_text(
            result.stdout, encoding="utf-8"
        )
        (self.root / f"command-{self.sequence:04d}.stderr.txt").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(
                f"browser {' '.join(arguments[:2])} failed: "
                + (result.stderr or result.stdout)[:300]
            )
        try:
            reply = json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError("browser returned invalid JSON") from exc
        if not isinstance(reply, dict) or reply.get("success") is not True:
            raise RuntimeError("browser did not confirm command completion")
        return reply.get("data")

    def start(self, url):
        target = urlsplit(url)
        if (
            target.scheme != "http"
            or target.hostname != "127.0.0.1"
            or target.port is None
            or target.username
            or target.password
            or target.path != "/screen"
            or target.query
            or target.fragment
        ):
            raise ValueError("screen capture requires the explicit local loopback screen")
        if self.started:
            raise ValueError("browser capture already started")
        self.started = True
        self.command("open", url)
        self.command("set", "viewport", "1920", "1080", "1")
        self.started_at = time.monotonic()
        self.command("record", "start", str(self.video), "--fps", str(SCREEN_FPS))
        self.recording = True

    def snapshot(self):
        return self.command("snapshot", "-i")

    def evaluate(self, expression):
        # Callers use read-only DOM observations, never injected recommendations.
        return self.command("eval", expression)

    def close(self):
        failures = []
        if self.recording:
            self.stop_started_at = time.monotonic()
            try:
                self.recording_receipt = self.command("record", "stop")
                self.recording = False
            except Exception as exc:
                failures.append(exc)
        if self.started:
            try:
                self.command("close")
            except Exception as exc:
                failures.append(exc)
        for name, identity in self.identities.items():
            if pin(getattr(self, name)) != identity:
                failures.append(RuntimeError(f"capture tool changed: {name}"))
        self.closed = not failures
        if failures:
            raise RuntimeError("capture stop/cleanup failed: " + "; ".join(map(str, failures)))
