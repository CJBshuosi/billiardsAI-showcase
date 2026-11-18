import hashlib
import json
from pathlib import Path

import pytest

from scripts.algorithm import package_recording_first_windows as builder
from scripts.algorithm.recording_packet import pin_file


def _stage(tmp_path):
    stage = tmp_path / "stage"
    packet = stage / "packet"
    packet.mkdir(parents=True)
    (packet / "source.py").write_bytes(b"original source\n")
    (packet / "prepare.ps1").write_bytes(b"original preparation\n")
    members = {name: pin_file(packet / name) for name in ("source.py", "prepare.ps1")}
    raw = json.dumps(
        dict(contract="recording-first-windows-packet/v1", source_commit="a" * 40, members=members)
    ).encode()
    (packet / "packet-manifest.json").write_bytes(raw)
    return stage, hashlib.sha256(raw).hexdigest()


def test_freeze_does_not_follow_mutated_stage_pins(tmp_path, monkeypatch):
    stage, digest = _stage(tmp_path)
    original = builder.zipfile.ZipFile.write

    def changed_write(archive, filename, arcname=None, *args, **kwargs):
        if arcname == "source.py":
            Path(filename).write_bytes(b"mutated after initial verification\n")
        return original(archive, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(builder.zipfile.ZipFile, "write", changed_write)
    frozen = tmp_path / "frozen"
    with pytest.raises(ValueError, match="frozen ZIP content"):
        builder.freeze_stage(stage, frozen, digest)
    assert not (frozen / "FROZEN.json").exists()


def test_external_preparation_matches_verified_zip_even_if_stage_changes(tmp_path, monkeypatch):
    stage, digest = _stage(tmp_path)
    original = Path.read_bytes

    def changed_read(path):
        if path == stage / "packet/prepare.ps1":
            path.write_bytes(b"late mutation\n")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    frozen = tmp_path / "frozen"
    try:
        archive_path = builder.freeze_stage(stage, frozen, digest)
    except ValueError:
        assert not (frozen / "FROZEN.json").exists()
        return
    with builder.zipfile.ZipFile(archive_path) as archive:
        assert (frozen / "prepare.ps1").read_bytes() == archive.read("prepare.ps1")


def _core_evidence_repo(tmp_path, monkeypatch, *, wrong_identity=None, change_core=False):
    import subprocess

    repo = tmp_path / "source-repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    def commit():
        git("add", ".")
        git(
            "-c",
            "user.name=Packet Test",
            "-c",
            "user.email=packet@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "packet fixture",
        )
        return git("rev-parse", "HEAD")

    git("init", "-q")
    for name in (*builder.HELPERS, "src/core.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'"""Frozen fixture source."""\n')
    baseline = commit()
    proof = dict(
        contract="durable-recording-d4-local-evidence/v1",
        source_commit=baseline,
        source_report_sha256="a" * 64,
        result=dict(product_pass=True, cleanup_confirmed=False),
        episode_pass=False,
        browser=dict(local_browser_pass=True, source_commit=baseline),
        tested_source_files={
            "src/core.py": hashlib.sha256((repo / "src/core.py").read_bytes()).hexdigest()
        },
    )
    if wrong_identity == "core":
        proof["source_commit"] = "b" * 40
    elif wrong_identity == "browser":
        proof["browser"]["source_commit"] = "b" * 40
    # Keep both historical and current evidence locations to expose identity
    # validation rather than failing simply because a filename changed.
    for name in (
        "recording-first-r4-local-evidence-2026-11-09.json",
        "durable-recording-d4-local-evidence-2026-11-10.json",
        Path(builder.LOCAL_EVIDENCE).name,
    ):
        path = repo / "docs/operations" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(proof))
    if change_core:
        (repo / "src/core.py").write_bytes(b'"""Unverified product change."""\n')
    revision = commit()
    monkeypatch.setattr(builder, "ROOT", repo)
    monkeypatch.setattr(builder, "PASSED_COMMIT", baseline)
    monkeypatch.setattr(builder, "PASSED_REPORT_SHA", "a" * 64)
    return revision


@pytest.mark.parametrize("wrong_identity", ["core", "browser"])
def test_packet_refuses_evidence_bound_to_another_core_revision(
    tmp_path, monkeypatch, wrong_identity
):
    revision = _core_evidence_repo(tmp_path, monkeypatch, wrong_identity=wrong_identity)
    with pytest.raises(ValueError, match="evidence identity"):
        builder._source(revision)


def test_packet_core_evidence_keeps_failed_episode_cleanup_visible(tmp_path, monkeypatch):
    revision = _core_evidence_repo(tmp_path, monkeypatch)
    source, blobs, proof = builder._source(revision)
    assert source["src/core.py"] == b'"""Frozen fixture source."""\n'
    assert "src/core.py" in blobs
    assert proof["result"]["cleanup_confirmed"] is False
    assert proof["episode_pass"] is False


def test_packet_cannot_reuse_core_evidence_after_product_change(tmp_path, monkeypatch):
    revision = _core_evidence_repo(tmp_path, monkeypatch, change_core=True)
    with pytest.raises(ValueError, match="product source changed"):
        builder._source(revision)
