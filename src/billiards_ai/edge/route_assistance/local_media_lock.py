"""Bounded cross-process session pins; control files are never media objects.

Readers and generators share a pin. Completion and collectors take an exclusive
pin before any database write. A SQLite lease is never a substitute for this lock.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import math
import os
import re
import stat
import time
from contextlib import contextmanager
from pathlib import Path


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", ctypes.c_uint32),
        ("OffsetHigh", ctypes.c_uint32),
        ("hEvent", ctypes.c_void_p),
    ]


def _windows_api():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.LockFileEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(_Overlapped),
    ]
    api.LockFileEx.restype = ctypes.c_int
    api.UnlockFileEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(_Overlapped),
    ]
    api.UnlockFileEx.restype = ctypes.c_int
    api.GetLastError.restype = ctypes.c_uint32
    api.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    api.CreateFileW.restype = ctypes.c_void_p
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    api.CloseHandle.restype = ctypes.c_int
    api.GetFileInformationByHandleEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    api.GetFileInformationByHandleEx.restype = ctypes.c_int
    return api


def _windows_open(api, path: Path, *, directory: bool):
    # No FILE_SHARE_DELETE: directories and the control file cannot be renamed
    # out from under the pin. OPEN_REPARSE_POINT prevents following the final hop.
    handle = api.CreateFileW(
        str(path),
        0x80000000 | (0 if directory else 0x40000000),
        3,
        None,
        3 if directory else 4,
        0x00200000 | (0x02000000 if directory else 0x80),
        None,
    )
    if handle in {None, ctypes.c_void_p(-1).value}:
        raise OSError(api.GetLastError(), "session media control open failed")
    attributes = (ctypes.c_uint32 * 2)()
    if (
        not api.GetFileInformationByHandleEx(
            handle, 9, ctypes.byref(attributes), ctypes.sizeof(attributes)
        )
        or attributes[0] & 0x400
        or bool(attributes[0] & 0x10) != directory
    ):
        api.CloseHandle(handle)
        raise OSError("session media control reparse point or type is unsafe")
    return handle


def _windows_try_lock(api, handle: int, *, exclusive: bool):
    overlapped = _Overlapped()
    # LOCKFILE_FAIL_IMMEDIATELY | optional LOCKFILE_EXCLUSIVE_LOCK.
    if api.LockFileEx(handle, 1 | (2 if exclusive else 0), 0, 1, 0, ctypes.byref(overlapped)):
        return overlapped
    error = api.GetLastError()
    if error == 33:  # ERROR_LOCK_VIOLATION is the only retryable condition.
        return None
    raise OSError(error, "session media lock failed")


def _windows_unlock(api, handle: int, overlapped) -> None:
    if not api.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlapped)):
        raise OSError(api.GetLastError(), "session media unlock failed")


def _assert_directory_chain(path: Path) -> None:
    for parent in (*reversed(path.parents), path):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("session media control root is unsafe")


@contextmanager
def session_media_lock(root: Path, session_id: str, *, exclusive: bool, timeout_seconds: float = 0):
    """Pin ``root/.session-media-locks/<sha256(session_id)>.lock``.

    Root must be the configured capture artifact root, shared by server and
    worker (store.artifact_path_for_session). No unlocked platform fallback.
    Pin acquisition/release must wrap the entire stream or media operation.
    """
    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,256}", session_id) is None
        or session_id in {".", ".."}
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
        or type(exclusive) is not bool
    ):
        raise ValueError("session media pin arguments are invalid")
    _assert_directory_chain(root)
    control = root / ".session-media-locks"
    control.mkdir(exist_ok=True, mode=0o700)
    _assert_directory_chain(control)
    name = hashlib.sha256(session_id.encode()).hexdigest() + ".lock"
    if os.name == "posix":
        import fcntl

        directory = os.open(control, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            descriptor = os.open(
                name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory
            )
        finally:
            os.close(directory)
        api = None
    elif os.name == "nt":
        import msvcrt

        api = _windows_api()
        directory_handles = []
        try:
            for parent in (*reversed(control.parents), control):
                directory_handles.append(_windows_open(api, parent, directory=True))
            handle = _windows_open(api, control / name, directory=False)
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
            except BaseException:
                api.CloseHandle(handle)
                raise
        except BaseException:
            for handle in reversed(directory_handles):
                api.CloseHandle(handle)
            raise
    else:
        raise OSError("session media locking is unsupported")
    acquired = False
    overlapped = None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("session media control file is unsafe")
        deadline = time.monotonic() + timeout_seconds
        while True:
            if api is None:
                try:
                    fcntl.flock(
                        descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
                    )
                    acquired = True
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
            else:
                overlapped = _windows_try_lock(
                    api, msvcrt.get_osfhandle(descriptor), exclusive=exclusive
                )
                acquired = overlapped is not None
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("session media is pinned")
            time.sleep(min(0.025, max(0, deadline - time.monotonic())))
        yield
    finally:
        try:
            if acquired:
                if api is None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                else:
                    _windows_unlock(api, msvcrt.get_osfhandle(descriptor), overlapped)
        finally:
            os.close(descriptor)
            if api is not None:
                for handle in reversed(directory_handles):
                    api.CloseHandle(handle)
