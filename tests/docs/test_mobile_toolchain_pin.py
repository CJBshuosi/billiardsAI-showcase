"""Executable contract for the pinned mobile Flutter toolchain."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_flutter.sh"
FLUTTER_VERSION = "3.44.7"
ARCHIVE = "stable/macos/flutter_macos_arm64_3.44.7-stable.zip"
STABLE_MACHINE_FIELDS = {
    "frameworkVersion",
    "channel",
    "repositoryUrl",
    "frameworkRevision",
    "frameworkCommitDate",
    "engineRevision",
    "engineCommitDate",
    "engineContentHash",
    "engineBuildDate",
    "dartSdkVersion",
    "devToolsVersion",
}


def _write_fake_archive(path: Path) -> str:
    """Return the archive's SHA-256 digest."""
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("flutter/bin/flutter")
        info.external_attr = 0o755 << 16
        archive.writestr(info, "#!/usr/bin/env sh\necho fake flutter\n")
    contents = path.read_bytes()
    return hashlib.sha256(contents).hexdigest()


def _fake_uname(path: Path, operating_system: str = "Darwin", architecture: str = "arm64") -> None:
    path.write_text(
        f'#!/usr/bin/env sh\ncase "$1" in\n  -s) echo {operating_system} ;;\n'
        f"  -m) echo {architecture} ;;\nesac\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_metadata(path: Path, sha256: str, archive: str = ARCHIVE) -> Path:
    payload = {
        "base_url": path.parent.joinpath("archives").as_uri(),
        "releases": [
            {
                "version": FLUTTER_VERSION,
                "channel": "stable",
                "archive": archive,
                "hash": "a" * 40,
                "sha256": sha256,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_manifest(path: Path, metadata: Path, sha256: str) -> Path:
    payload = {
        "schema_version": 1,
        "flutter_version": FLUTTER_VERSION,
        "channel": "stable",
        "host_os": "macos",
        "host_arch": "arm64",
        "archive": ARCHIVE,
        "official_release_url": f"{path.parent.joinpath('archives').as_uri()}/{ARCHIVE}",
        "official_release_json_url": metadata.as_uri(),
        "official_release_hash": "a" * 40,
        "official_archive_sha256": sha256,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_bootstrap(
    tmp_path: Path,
    metadata: Path,
    manifest: Path,
    cache: Path,
    *,
    operating_system: str = "Darwin",
    architecture: str = "arm64",
) -> subprocess.CompletedProcess[str]:
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    _fake_uname(tools / "uname", operating_system, architecture)
    environment = os.environ | {
        "FLUTTER_RELEASES_URL": metadata.as_uri(),
        "FLUTTER_RELEASE_MANIFEST": str(manifest),
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        [str(BOOTSTRAP), str(cache)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _verified_fixture(tmp_path: Path, *, sha256: str | None = None) -> tuple[Path, Path, Path, str]:
    archive = tmp_path / "archives" / ARCHIVE
    actual_sha256 = _write_fake_archive(archive)
    metadata = _write_metadata(tmp_path / "releases_macos.json", actual_sha256)
    manifest = _write_manifest(
        tmp_path / "flutter_release_manifest.json",
        metadata,
        actual_sha256 if sha256 is None else sha256,
    )
    return metadata, manifest, archive, actual_sha256


def test_bootstrap_selects_the_pinned_macos_arm64_archive_into_caller_cache(tmp_path: Path) -> None:
    """The verified macOS arm64 SDK is installed only under the requested cache."""
    metadata, manifest, _, _ = _verified_fixture(tmp_path)
    cache = tmp_path / "outside-repository-cache"

    completed = _run_bootstrap(tmp_path, metadata, manifest, cache)

    binary = cache / "flutter-3.44.7-macos-arm64" / "flutter" / "bin" / "flutter"
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(binary)
    assert binary.is_file()
    assert REPO_ROOT not in binary.parents


def test_bootstrap_repairs_a_tampered_cached_binary_from_the_verified_archive(
    tmp_path: Path,
) -> None:
    """Every run rebuilds the SDK instead of trusting a cached executable."""
    metadata, manifest, _, _ = _verified_fixture(tmp_path)
    cache = tmp_path / "outside-repository-cache"

    first = _run_bootstrap(tmp_path, metadata, manifest, cache)
    binary = cache / "flutter-3.44.7-macos-arm64" / "flutter" / "bin" / "flutter"
    assert first.returncode == 0, first.stderr
    binary.write_text("tampered\n", encoding="utf-8")

    second = _run_bootstrap(tmp_path, metadata, manifest, cache)

    assert second.returncode == 0, second.stderr
    assert binary.read_text(encoding="utf-8") == "#!/usr/bin/env sh\necho fake flutter\n"


def test_bootstrap_fails_closed_when_official_metadata_sha256_drifts_from_manifest(
    tmp_path: Path,
) -> None:
    """A changed official metadata SHA-256 is rejected before downloading the archive."""
    metadata, manifest, _, _ = _verified_fixture(tmp_path)
    changed = json.loads(metadata.read_text(encoding="utf-8"))
    changed["releases"][0]["sha256"] = "0" * 64
    metadata.write_text(json.dumps(changed), encoding="utf-8")

    completed = _run_bootstrap(tmp_path, metadata, manifest, tmp_path / "outside-repository-cache")

    assert completed.returncode != 0
    assert "official metadata SHA-256 does not match manifest" in completed.stderr


def test_bootstrap_rejects_bytes_when_only_release_hash_matches(
    tmp_path: Path,
) -> None:
    """The 40-character release hash cannot authorize archive bytes."""
    metadata, manifest, archive, _ = _verified_fixture(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tampered")

    completed = _run_bootstrap(tmp_path, metadata, manifest, tmp_path / "outside-repository-cache")

    assert completed.returncode != 0
    assert "official SHA-256 verification failed" in completed.stderr


def test_bootstrap_requires_exact_official_sha256_in_manifest(tmp_path: Path) -> None:
    """The manifest must pin the exact SHA-256 supplied by official metadata."""
    metadata, manifest, _, _ = _verified_fixture(tmp_path, sha256="f" * 64)

    completed = _run_bootstrap(tmp_path, metadata, manifest, tmp_path / "outside-repository-cache")

    assert completed.returncode != 0
    assert "official metadata SHA-256 does not match manifest" in completed.stderr


def test_bootstrap_rejects_wrong_host_or_archive(tmp_path: Path) -> None:
    """Only the approved macOS arm64 archive is eligible for this bootstrap."""
    metadata, manifest, _, _ = _verified_fixture(tmp_path)

    wrong_host = _run_bootstrap(
        tmp_path,
        metadata,
        manifest,
        tmp_path / "cache-one",
        operating_system="Linux",
        architecture="x86_64",
    )
    assert wrong_host.returncode != 0
    assert "Unsupported Flutter host" in wrong_host.stderr

    wrong_archive_metadata = _write_metadata(
        tmp_path / "wrong-archive.json", "0" * 40, "stable/macos/flutter_macos_3.44.7-stable.zip"
    )
    manifest_contents = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_contents["official_release_json_url"] = wrong_archive_metadata.as_uri()
    manifest.write_text(json.dumps(manifest_contents), encoding="utf-8")
    wrong_archive = _run_bootstrap(
        tmp_path, wrong_archive_metadata, manifest, tmp_path / "cache-two"
    )
    assert wrong_archive.returncode != 0
    assert "official metadata archive does not match manifest" in wrong_archive.stderr


def test_bootstrap_rejects_a_cache_inside_the_repository(tmp_path: Path) -> None:
    """The caller cannot accidentally put the SDK under Git's working tree."""
    metadata, manifest, _, _ = _verified_fixture(tmp_path)
    cache = REPO_ROOT / ".flutter-sdk-cache"
    assert not cache.exists(), "test cache must not overwrite pre-existing local state"

    completed = _run_bootstrap(tmp_path, metadata, manifest, cache)

    assert completed.returncode != 0
    assert "outside the repository" in completed.stderr
    assert not cache.exists()


def test_concurrent_bootstrap_rejects_second_writer_before_curl(tmp_path: Path) -> None:
    """An active bootstrap lock prevents a second process from touching shared downloads."""
    metadata, manifest, _, _ = _verified_fixture(tmp_path)
    cache = tmp_path / "outside-repository-cache"
    tools = tmp_path / "tools"
    tools.mkdir()
    _fake_uname(tools / "uname")
    curl_log = tmp_path / "curl.log"
    archive_started = tmp_path / "archive-started"
    curl_wrapper = tools / "curl"
    curl_wrapper.write_text(
        "#!/usr/bin/env sh\n"
        'printf \'%s\\n\' "$*" >> "$CURL_LOG"\n'
        'case "$*" in *flutter_macos_arm64_3.44.7-stable.zip*)\n'
        '  touch "$ARCHIVE_STARTED"; sleep 2 ;;\n'
        "esac\n"
        'exec "$CURL_REAL" "$@"\n',
        encoding="utf-8",
    )
    curl_wrapper.chmod(curl_wrapper.stat().st_mode | stat.S_IXUSR)
    environment = os.environ | {
        "FLUTTER_RELEASES_URL": metadata.as_uri(),
        "FLUTTER_RELEASE_MANIFEST": str(manifest),
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "CURL_REAL": shutil.which("curl") or "",
        "CURL_LOG": str(curl_log),
        "ARCHIVE_STARTED": str(archive_started),
    }
    first = subprocess.Popen(
        [str(BOOTSTRAP), str(cache)],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not archive_started.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert archive_started.exists(), first.communicate(timeout=5)[1]
    curl_log_before_second = curl_log.read_text(encoding="utf-8")

    second = subprocess.run(
        [str(BOOTSTRAP), str(cache)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    first_stdout, first_stderr = first.communicate(timeout=10)

    assert first.returncode == 0, first_stderr
    assert first_stdout.strip().endswith("flutter/bin/flutter")
    assert second.returncode != 0
    assert "another Flutter bootstrap is already running" in second.stderr
    assert curl_log.read_text(encoding="utf-8") == curl_log_before_second


def test_checked_in_pin_records_the_exact_stable_release_and_machine_output() -> None:
    """The committed manifest and toolchain output pin the approved exact release."""
    manifest = json.loads(
        (REPO_ROOT / "apps" / "mobile" / "flutter_release_manifest.json").read_text()
    )
    toolchain = json.loads((REPO_ROOT / "apps" / "mobile" / "toolchain.json").read_text())

    assert manifest["flutter_version"] == FLUTTER_VERSION
    assert manifest["channel"] == "stable"
    assert manifest["host_os"] == "macos"
    assert manifest["host_arch"] == "arm64"
    assert manifest["archive"] == ARCHIVE
    assert len(manifest["official_release_hash"]) == 40
    assert len(manifest["official_archive_sha256"]) == 64
    assert set(toolchain) == STABLE_MACHINE_FIELDS
    assert toolchain["frameworkVersion"] == FLUTTER_VERSION
    assert toolchain["channel"] == "stable"
    assert toolchain["frameworkRevision"] == manifest["official_release_hash"]
    assert toolchain["repositoryUrl"] == "https://github.com/flutter/flutter.git"
    assert toolchain["dartSdkVersion"] == "3.12.2"
    assert toolchain["devToolsVersion"] == "2.57.0"
    assert "flutterRoot" not in toolchain
    assert not any(value.startswith("/") for value in toolchain.values() if isinstance(value, str))
    assert ".superpowers/sdd/**/flutter-version-machine.raw.json" in (
        REPO_ROOT / ".gitignore"
    ).read_text(encoding="utf-8")
