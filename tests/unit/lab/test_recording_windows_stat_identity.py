"""CPython Windows path stat reports birth time where fd stat reports change time."""

import hashlib
import os
import threading
from pathlib import Path

import pytest

from billiards_ai.lab import mainline1_recording_media as media
from billiards_ai.lab.mainline1_realtime_ring import RealtimeRingError
from billiards_ai.lab.mainline1_session_recording import CommittedRecordingSource
from tests.unit.lab.test_mainline1_realtime_ring import _frame
from tests.unit.lab.test_recording_committed_source import _recorder


class _WindowsStat:
    """Preserve real file metadata except the documented path/fd clock difference."""

    def __init__(self, real, *, by_path):
        self._real = real
        self.st_birthtime_ns = 123456789
        self.st_ctime_ns = self.st_birthtime_ns if by_path else real.st_ctime_ns

    def __getattr__(self, key):
        return getattr(self._real, key)


def _windows_stat_views(monkeypatch, target, *, mutate_after_read, path_stat=False):
    original_lstat, original_fstat, original_stat = Path.lstat, os.fstat, Path.stat
    original = target.stat()
    content = target.read_bytes()
    calls = 0

    def lstat(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        return _WindowsStat(info, by_path=True) if path == target else info

    def fstat(fd):
        nonlocal calls
        info = original_fstat(fd)
        if (info.st_dev, info.st_ino) != (original.st_dev, original.st_ino):
            return info
        calls += 1
        if mutate_after_read and calls == 2:
            # Keep inode, byte count and mtime; fd change time must still catch
            # a mutation after the first hash/read and before final validation.
            target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
            info = original_fstat(fd)
        return _WindowsStat(info, by_path=False)

    if path_stat:

        def stat(path, *args, **kwargs):
            info = original_stat(path, *args, **kwargs)
            return _WindowsStat(info, by_path=True) if path == target else info

        monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(os, "fstat", fstat)


@pytest.mark.parametrize("mutate", [False, True])
def test_first_committed_segment_handles_windows_stat_clocks_without_losing_fd_guards(
    tmp_path, monkeypatch, mutate
):
    recorder = _recorder(tmp_path)
    originals = [_frame(index) for index in range(60)]
    for frame in originals:
        recorder.append(frame)
    manifest = recorder.close()
    target = tmp_path / "raw" / manifest.segments[0].name
    source = CommittedRecordingSource(
        recorder=recorder, session_id="session-1", source_epoch_id="epoch-1"
    )
    _windows_stat_views(monkeypatch, target, mutate_after_read=mutate)
    try:
        if mutate:
            with pytest.raises(RealtimeRingError, match="hash changed"):
                list(source.frames(threading.Event()))
            assert source.progress.last_delivered_frame_id is None
        else:
            assert list(source.frames(threading.Event())) == originals
            assert source.progress.last_delivered_frame_id == 59
    finally:
        source.cancel()


@pytest.mark.parametrize("mutate", [False, True])
def test_media_reference_handles_windows_stat_clocks_without_losing_fd_guards(
    tmp_path, monkeypatch, mutate
):
    target = tmp_path / "reference.json"
    original = b'{"capture":"original"}\n'
    target.write_bytes(original)
    _windows_stat_views(monkeypatch, target, mutate_after_read=mutate)
    if mutate:
        with pytest.raises(media.RecordingPlaybackError, match="metadata changed while reading"):
            media._read_safe_small_file(target, hashlib.sha256(original).hexdigest())
    else:
        assert media._read_safe_small_file(target, hashlib.sha256(original).hexdigest()) == original
