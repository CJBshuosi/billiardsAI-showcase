from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest


def _try_pin(root: str, exclusive: bool, result) -> None:
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    try:
        with session_media_lock(Path(root), "session", exclusive=exclusive, timeout_seconds=0):
            result.put("acquired")
    except TimeoutError:
        result.put("busy")


def test_session_pins_fence_other_process_and_release_on_exception(tmp_path: Path):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with pytest.raises(RuntimeError):
        with session_media_lock(tmp_path, "session", exclusive=False, timeout_seconds=0):
            for exclusive, expected in ((False, "acquired"), (True, "busy")):
                child = context.Process(target=_try_pin, args=(str(tmp_path), exclusive, results))
                child.start()
                child.join(10)
                assert child.exitcode == 0
                assert results.get(timeout=2) == expected
            raise RuntimeError("cancelled")
    with session_media_lock(tmp_path, "session", exclusive=True, timeout_seconds=0):
        child = context.Process(target=_try_pin, args=(str(tmp_path), False, results))
        child.start()
        child.join(10)
        assert child.exitcode == 0
        assert results.get(timeout=2) == "busy"
    assert (tmp_path / ".session-media-locks").is_dir()


def test_pin_refuses_symlink_control_root_and_invalid_identity(tmp_path: Path):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".session-media-locks").symlink_to(outside, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        with session_media_lock(tmp_path, "session", exclusive=True, timeout_seconds=0):
            pytest.fail("unsafe pin acquired")
    with pytest.raises(ValueError):
        with session_media_lock(tmp_path, "../session", exclusive=True, timeout_seconds=0):
            pytest.fail("unsafe identity acquired")


def test_windows_native_lock_contract_has_shared_exclusive_and_no_unlocked_errors():
    from billiards_ai.edge.route_assistance.local_media_lock import (
        _windows_try_lock,
        _windows_unlock,
    )

    class Kernel:
        def __init__(self):
            self.success, self.error, self.calls = 1, 0, []

        def LockFileEx(self, *args):
            self.calls.append(args[:5])
            return self.success

        def UnlockFileEx(self, *args):
            self.calls.append(args[:4])
            return self.success

        def GetLastError(self):
            return self.error

    kernel = Kernel()
    shared = _windows_try_lock(kernel, 123, exclusive=False)
    assert kernel.calls[-1] == (123, 1, 0, 1, 0)
    exclusive = _windows_try_lock(kernel, 123, exclusive=True)
    assert kernel.calls[-1] == (123, 3, 0, 1, 0)
    assert shared is not None and exclusive is not None
    _windows_unlock(kernel, 123, exclusive)
    assert kernel.calls[-1] == (123, 0, 1, 0)
    kernel.success, kernel.error = 0, 33
    assert _windows_try_lock(kernel, 123, exclusive=True) is None
    kernel.error = 5
    with pytest.raises(OSError):
        _windows_try_lock(kernel, 123, exclusive=True)
    with pytest.raises(OSError):
        _windows_unlock(kernel, 123, shared)


def test_windows_control_handles_deny_delete_share_and_reparse_points(tmp_path: Path):
    import ctypes

    from billiards_ai.edge.route_assistance.local_media_lock import _windows_open

    class Kernel:
        def __init__(self):
            self.attributes, self.closed, self.arguments = 0x80, [], ()

        def CreateFileW(self, *args):
            self.arguments = args
            return 123

        def GetFileInformationByHandleEx(self, handle, kind, buffer, size):
            assert handle == 123 and kind == 9 and size == 8
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint32))[0] = self.attributes
            return 1

        def CloseHandle(self, handle):
            self.closed.append(handle)

    kernel = Kernel()
    assert _windows_open(kernel, tmp_path / "pin", directory=False) == 123
    assert kernel.arguments[2] == 3  # FILE_SHARE_READ | FILE_SHARE_WRITE, never DELETE.
    assert kernel.arguments[4] == 4  # OPEN_ALWAYS, never truncate an active pin.
    assert kernel.arguments[5] & 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT.
    kernel.attributes = 0x410  # directory + reparse point
    with pytest.raises(OSError, match="unsafe"):
        _windows_open(kernel, tmp_path, directory=True)
    assert kernel.closed == [123]


def _hold_until_terminated(root: str, ready) -> None:
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    with session_media_lock(Path(root), "session", exclusive=False):
        ready.set()
        ready = multiprocessing.Event()
        ready.wait(20)


def test_process_death_releases_pin_without_lease_expiration(tmp_path: Path):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    child = context.Process(target=_hold_until_terminated, args=(str(tmp_path), ready))
    child.start()
    try:
        assert ready.wait(10)
        with pytest.raises(TimeoutError):
            with session_media_lock(tmp_path, "session", exclusive=True, timeout_seconds=0.01):
                pytest.fail("live process pin lost")
    finally:
        child.terminate()
        child.join(10)
    with session_media_lock(tmp_path, "session", exclusive=True):
        pass
