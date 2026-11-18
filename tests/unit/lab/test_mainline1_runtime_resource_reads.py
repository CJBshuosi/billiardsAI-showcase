"""Read real bytes while controlling only platform-specific stat observations."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab import mainline1_runtime as runtime

_RAW = b'{"fixture":"private-resource-content"}'
_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


@pytest.fixture(params=["sha256", "json"])
def reader(request):
    if request.param == "sha256":
        return runtime._sha256
    return lambda path: runtime._json_object(path, label="fixture")


@pytest.fixture
def resource(tmp_path):
    path = tmp_path / "private-resource-name.json"
    path.write_bytes(_RAW)
    return path


def _controlled_stats(monkeypatch, path, *, platform="nt", on_read=None):
    """Keep actual open/read/stat and isolate modeled Windows fields to this module."""
    safe_stat = runtime._safe_stat
    fstat = os.fstat
    path_open = Path.open
    baseline = safe_stat(path)
    common = {field: getattr(baseline, field) for field in _FIELDS}
    common.update(st_mode=baseline.st_mode, st_birthtime_ns=50)
    state = SimpleNamespace(
        read=False,
        path_before={"st_ctime_ns": 100},
        fd_before={"st_ctime_ns": 200 if platform == "nt" else 100},
        path_after={},
        fd_after={},
    )

    def snapshot(actual, api):
        values = dict(common)
        values["st_mode"] = actual.st_mode
        values.update(getattr(state, f"{api}_before"))
        if state.read:
            values.update(getattr(state, f"{api}_after"))
        return SimpleNamespace(**{k: v for k, v in values.items() if v is not None})

    def checked_path_stat(target):
        actual = safe_stat(target)
        return snapshot(actual, "path") if target == path else actual

    def checked_fstat(fd):
        return snapshot(fstat(fd), "fd")

    class Reader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def fileno(self):
            return self.source.fileno()

        def read(self, size):
            result = self.source.read(size)
            if not state.read:
                state.read = True
                if on_read is not None:
                    on_read()
            return result

    def open_resource(target, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if target != path or mode != "rb":
            return path_open(target, *args, **kwargs)
        return Reader(path_open(target, *args, **kwargs))

    monkeypatch.setattr(runtime, "_STAT_PLATFORM", platform, raising=False)
    monkeypatch.setattr(runtime, "_safe_stat", checked_path_stat)
    monkeypatch.setattr(runtime, "os", SimpleNamespace(fstat=checked_fstat))
    monkeypatch.setattr(Path, "open", open_resource)
    return state


def test_stable_windows_creation_and_change_ctimes_accept_exact_bytes(
    monkeypatch, resource, reader
):
    _controlled_stats(monkeypatch, resource)
    result = reader(resource)
    expected = (
        hashlib.sha256(_RAW).hexdigest()
        if reader is runtime._sha256
        else {"fixture": "private-resource-content"}
    )
    assert result == expected


@pytest.mark.parametrize("field", [*_FIELDS[:4], "st_birthtime_ns"])
def test_cross_api_identity_disagreement_at_open_is_rejected(monkeypatch, resource, reader, field):
    state = _controlled_stats(monkeypatch, resource)
    state.fd_before[field] = -1
    with pytest.raises(ValueError) as caught:
        reader(resource)
    assert "before read" in str(caught.value)
    assert field in str(caught.value)


def test_posix_ctime_only_disagreement_at_open_is_rejected(monkeypatch, resource, reader):
    state = _controlled_stats(monkeypatch, resource, platform="posix")
    state.fd_before["st_ctime_ns"] = 200
    with pytest.raises(ValueError) as caught:
        reader(resource)
    assert "before read" in str(caught.value)
    assert "st_ctime_ns" in str(caught.value)


@pytest.mark.parametrize("api", ["path", "fd"])
@pytest.mark.parametrize("field", [*_FIELDS, "st_birthtime_ns"])
def test_same_api_mutation_after_read_is_rejected(monkeypatch, resource, reader, api, field):
    state = _controlled_stats(monkeypatch, resource)
    getattr(state, f"{api}_after")[field] = -1
    with pytest.raises(ValueError) as caught:
        reader(resource)
    assert ("after read" if api == "path" else "during read") in str(caught.value)
    assert field in str(caught.value)


@pytest.mark.parametrize("api", ["path", "fd"])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_missing_windows_birthtime_fails_closed(monkeypatch, resource, reader, api, phase):
    state = _controlled_stats(monkeypatch, resource)
    getattr(state, f"{api}_{phase}")["st_birthtime_ns"] = None
    with pytest.raises(ValueError, match="st_birthtime_ns"):
        reader(resource)


@pytest.mark.parametrize("phase", ["open", "read"])
def test_change_diagnostics_retain_phase_and_fields_without_paths_or_contents(
    monkeypatch, resource, reader, phase
):
    state = _controlled_stats(monkeypatch, resource)
    target = state.fd_before if phase == "open" else state.fd_after
    target.update(st_ino=-1, st_size=-1)
    with pytest.raises(ValueError) as caught:
        reader(resource)
    message = str(caught.value)
    assert ("before read" if phase == "open" else "during read") in message
    assert "st_ino" in message and "st_size" in message
    assert str(resource) not in message and resource.name not in message
    assert "private-resource-content" not in message


def test_full_file_identity_keeps_ctime():
    info = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5)
    assert runtime._file_identity(info) == (1, 2, 3, 4, 5)


@pytest.mark.parametrize("when", ["before_open", "during_read", "swap_restore"])
def test_real_filesystem_replacement_is_rejected(monkeypatch, resource, reader, when):
    path_open = Path.open
    before = resource.stat()
    replacement = resource.with_name("replacement.json")
    replacement.write_bytes(_RAW)
    moved = resource.with_name("moved.json")

    def replace():
        if when == "swap_restore":
            resource.rename(moved)
            replacement.replace(resource)
            moved.replace(resource)
            os.utime(resource, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
        else:
            replacement.replace(resource)

    class ReplacingReader:
        def __init__(self, source):
            self.source = source
            self.changed = False

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def fileno(self):
            return self.source.fileno()

        def read(self, size):
            result = self.source.read(size)
            if not self.changed:
                self.changed = True
                replace()
            return result

    def open_resource(target, *args, **kwargs):
        if target != resource:
            return path_open(target, *args, **kwargs)
        if when == "before_open":
            replace()
            return path_open(target, *args, **kwargs)
        return ReplacingReader(path_open(target, *args, **kwargs))

    monkeypatch.setattr(Path, "open", open_resource)
    with pytest.raises(ValueError):
        reader(resource)


def test_swap_restore_with_only_same_api_ctime_change_is_rejected(monkeypatch, resource, reader):
    moved = resource.with_name("moved.json")

    def swap_restore():
        resource.rename(moved)
        resource.write_bytes(_RAW)
        moved.replace(resource)

    state = _controlled_stats(monkeypatch, resource, on_read=swap_restore)
    state.path_after["st_ctime_ns"] = 101
    state.fd_after["st_ctime_ns"] = 201
    with pytest.raises(ValueError, match="st_ctime_ns"):
        reader(resource)


@pytest.mark.parametrize("raw", [b"", b"ordinary hash bytes", b"x" * (1024 * 1024 + 1)])
def test_hash_accepts_ordinary_empty_and_multichunk_content(resource, raw):
    resource.write_bytes(raw)
    assert runtime._sha256(resource) == hashlib.sha256(raw).hexdigest()


def test_json_accepts_exact_one_mib_boundary(resource):
    resource.write_bytes(b"{}" + b" " * (1024 * 1024 - 2))
    assert runtime._json_object(resource, label="fixture") == {}


@pytest.mark.parametrize("raw", [b"{}" + b" " * (1024 * 1024 - 1), b"{", b"[]", b"\xff"])
def test_json_rejects_oversized_malformed_nonobject_and_nonutf8_without_details(resource, raw):
    resource.write_bytes(raw)
    with pytest.raises(ValueError, match="^fixture is invalid$"):
        runtime._json_object(resource, label="fixture")
