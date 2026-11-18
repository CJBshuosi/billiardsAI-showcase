import importlib.util
import pathlib
import sqlite3

import pytest

ENTRY = (
    pathlib.Path(__file__).resolve().parents[3] / "scripts/development/blackbox_media_diagnostic.py"
)


def diagnostic():
    assert ENTRY.is_file(), "bounded component diagnostic has not been implemented"
    spec = importlib.util.spec_from_file_location("media_diagnostic_test", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backup_includes_uncheckpointed_wal_without_mutating_source(tmp_path):
    module = diagnostic()
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("CREATE TABLE sample (value INTEGER)")
        live.execute("INSERT INTO sample VALUES (42)")
        live.commit()
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        target = tmp_path / "copy.sqlite3"
        module.backup_database(source, target, lambda: None)
        with sqlite3.connect(target) as copy:
            assert copy.execute("SELECT value FROM sample").fetchall() == [(42,)]
        assert {name: (tmp_path / name).read_bytes() for name in before} == before
        with pytest.raises(FileExistsError):
            module.backup_database(source, target, lambda: None)


def test_profiler_preserves_exception_and_records_failed_stage():
    module = diagnostic()
    rows = []
    profile = module.Profiler(rows.append, seconds=60)

    def fail():
        raise ValueError("private path should not enter export")

    with pytest.raises(ValueError):
        profile.call("ENCODE", fail)
    assert rows[-1]["stage"] == "ENCODE_END"
    assert rows[-1]["complete"] is False
    assert "private" not in str(rows)


def test_profiler_does_not_skip_checks_and_keeps_partial_counts():
    module = diagnostic()
    rows, called = [], []
    profile = module.Profiler(rows.append, seconds=60)

    def check():
        called.append(True)
        if len(called) == 5:
            raise RuntimeError("stop")

    measured = profile.wrap("continue", check)
    for _ in range(4):
        measured()
    with pytest.raises(RuntimeError):
        measured()
    assert len(called) == 5
    assert profile.metrics["continue"]["count"] == 5
    assert profile.metrics["continue"]["errors"] == 1


def test_budget_rejects_before_starting_extra_work():
    module = diagnostic()
    clock = iter([0.0, 61.0]).__next__
    rows = []
    profile = module.Profiler(rows.append, seconds=60, clock=clock)
    with pytest.raises(TimeoutError):
        profile.check()
    assert rows == []


def test_export_omits_controller_exception_text():
    module = diagnostic()
    exported = module.public_owned(
        {
            "status": "FAIL",
            "reason": "exception",
            "exception": "C:/private/token=abc",
            "cleanup_confirmed": True,
            "worker_exit_code": None,
            "terminal_record_error": "private",
        }
    )
    assert exported["status"] == "FAIL"
    assert exported["cleanup_confirmed"] is True
    assert "private" not in str(exported)


def test_cleanup_measurement_runs_after_budget_exhaustion():
    module = diagnostic()
    rows, released = [], []
    profile = module.Profiler(rows.append, seconds=60, clock=lambda: 0)
    profile.started = -61
    release = profile.wrap("writer_release", lambda: released.append(True), check_budget=False)
    release()
    assert released == [True]
    assert profile.metrics["writer_release"]["count"] == 1


def test_path_check_rejects_link_before_resolving_descendant(tmp_path, monkeypatch):
    module = diagnostic()
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    path = link / "unreadable"
    original = pathlib.Path.resolve

    def guarded_resolve(self, *args, **kwargs):
        assert self != path, "descendant was traversed before rejecting parent link"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", guarded_resolve)
    with pytest.raises(ValueError):
        module.safe_under(path, root)


def test_event_export_discards_native_text_and_private_nested_fields():
    module = diagnostic()
    raw = (
        b'private native log\n{"stage":"PROGRESS","active_sta'
        b'ge":"PUBLISH_ONE","secret":"token","metrics":{"con'
        b'tinue":{"count":2,"seconds":0.5,"errors":0,"secret'
        b'":"path"},"private":{"count":1}}}\n{"stage":"FAILED'
        b'","exception_type":"ValueError","trace":"private"}'
        b'\n{"stage":'
    )
    rows = module.export_events(raw)
    assert len(rows) == 2
    assert rows[0]["metrics"] == {"continue": {"count": 2, "seconds": 0.5, "errors": 0}}
    assert "private" not in str(rows) and "token" not in str(rows)


def test_runtime_path_tampering_is_rejected(tmp_path):
    module = diagnostic()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    path = runtime / "python312._pth"
    path.write_bytes(module.expected_runtime_paths(tmp_path))
    module.verify_runtime_paths(tmp_path)
    path.write_bytes(b"python312.zip\nC:/wrong/source\n")
    with pytest.raises(ValueError, match="RUNTIME_PATH_IDENTITY"):
        module.verify_runtime_paths(tmp_path)


def test_event_export_handles_malformed_types_and_nonfinite_numbers():
    module = diagnostic()
    raw = (
        b'{"stage":[]}\n{"stage":"PROGRESS","active_stage":{}'
        b',"seconds":NaN,"metrics":{"continue":{"seconds":In'
        b'finity,"count":2}}}\n'
    )
    rows = module.export_events(raw)
    assert len(rows) == 1
    assert "seconds" not in rows[0]
    assert rows[0]["metrics"]["continue"] == {"count": 2}
