"""Explicit postreview disposal of registered copied media, without cloud access.

The durable journal precedes rename and unlink. Only its exact inventoried names
may recover ENOENT. Staging is private and on the original filesystem; neither
directories nor metadata are recursively removed. A lease never replaces a pin.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import sqlite3
import stat
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from billiards_ai.edge.route_assistance.local_media_lock import (
    _assert_directory_chain,
    _windows_api,
    session_media_lock,
)
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.owner_media_deletion import LocalOwnerDeleteAttempt
from billiards_ai.edge.route_assistance.postgame import session_scoped_clip_id
from billiards_ai.edge.route_assistance.retention_inventory import _identity, _job, _json, _sha
from billiards_ai.lab.mainline1_recording_media import (
    RecordingPlaybackError,
    recording_playback_inventory,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    LegacyFrozenJobMediaSourceV1,
    ShotMediaEncodeJobV1,
    _media_job_from_payload,
)

_MAX_OBJECTS = 64
_MAX_ARTIFACT_ENTRIES = 16384
_MAX_JSON = 1024 * 1024
_CHUNK = 1024 * 1024
_MAX_MEDIA_BYTES = 8 * 1024**3
_FACT_NAMES = frozenset(
    {
        "active-tracks.json",
        "anchors.json",
        "cue-track.json",
        "detections.json",
        "event-timeline.json",
        "events.json",
        "motion-fragments.json",
        "performance.json",
        "shot-windows.json",
        "summary.json",
        "table-state-after.json",
        "table-state-before.json",
        "world-events.json",
    }
)


@dataclass(frozen=True, slots=True)
class LocalDiscardAttempt:
    outcome: str
    error_code: str | None = None
    reclaimed_bytes: int = 0
    session_id: str | None = None
    shot_id: str | None = None


class _Stale(Exception):
    pass


class _Unsafe(ValueError):
    pass


def _canonical(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode()) > 64 * 1024:
        raise _Unsafe("journal_limit")
    return encoded


def _absolute(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or ".project-archive" in path.parts:
        raise _Unsafe("unsafe_path")
    return path


def _name(value):
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or not value
        or Path(value).name != value
        or any(c in value for c in "\\/:\x00")
    ):
        raise _Unsafe("unsafe_name")
    return value


def _identity_of(info):
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or getattr(info, "st_file_attributes", 0) & 0x400
        or not info.st_ino
    ):
        raise _Unsafe("unsafe_object")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


class _PosixFS:
    """All POSIX mutations are relative to no-follow, retained descriptors."""

    @contextmanager
    def directory(self, path):
        with ExitStack() as stack:
            descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, descriptor)
            for part in path.parts[1:]:
                descriptor = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                stack.callback(os.close, descriptor)
            yield descriptor

    def names(self, directory, path, *, limit=_MAX_OBJECTS):
        names = []
        with os.scandir(directory) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > limit:
                    raise _Unsafe("directory_limit")
        return set(names)

    @contextmanager
    def file(self, directory, path, *, mutable=False):
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        try:
            _identity_of(os.fstat(descriptor))
            yield descriptor
        finally:
            os.close(descriptor)

    def mkdir(self, directory, path):
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=directory)
            os.fsync(directory)
        except FileExistsError:
            pass

    def private(self, directory):
        info = os.fstat(directory)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise _Unsafe("unsafe_staging_permissions")

    def device(self, directory):
        return os.fstat(directory).st_dev

    def rename(self, source, source_path, target, target_path, descriptor):
        # Never replace an unexpected object, even if it appeared after inventory.
        _rename_no_replace(source, source_path.name, target, target_path.name)
        os.fsync(source)
        os.fsync(target)

    def unlink(self, directory, path, descriptor):
        current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if _identity_of(current) != _identity_of(os.fstat(descriptor)):
            raise _Unsafe("changed_object")
        os.unlink(path.name, dir_fd=directory)
        os.fsync(directory)


def _rename_no_replace(source, name, target, destination):
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        native, flag = getattr(library, "renameatx_np", None), 4  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        native, flag = getattr(library, "renameat2", None), 1  # RENAME_NOREPLACE
    else:
        raise _Unsafe("atomic_no_replace_unsupported")
    if native is None:
        raise _Unsafe("atomic_no_replace_unsupported")
    native.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    native.restype = ctypes.c_int
    if native(source, os.fsencode(name), target, os.fsencode(destination), flag):
        error = ctypes.get_errno()
        raise OSError(error, "detach_failed")


def _windows_existing(api, path, *, directory, mutable=False):
    # Existing only; no FILE_SHARE_DELETE, including every ancestor directory.
    # Metadata/DB reads must coexist with SQLite VFS opens that omit SHARE_DELETE.
    # Request DELETE access only for exact media handles used for rename/unlink.
    handle = api.CreateFileW(
        str(path),
        0x80 if directory else 0x80000000 | (0x10000 if mutable else 0),
        3,
        None,
        3,
        0x00200000 | (0x02000000 if directory else 0x80),
        None,
    )
    if handle in {None, ctypes.c_void_p(-1).value}:
        error = api.GetLastError()
        if error in {2, 3}:
            raise FileNotFoundError(errno.ENOENT, "media_missing")
        if error in {5, 32, 33}:
            raise PermissionError(errno.EACCES, "media_busy")
        raise OSError(error, "media_open_failed")
    attributes = (ctypes.c_uint32 * 2)()
    if (
        not api.GetFileInformationByHandleEx(
            handle, 9, ctypes.byref(attributes), ctypes.sizeof(attributes)
        )
        or attributes[0] & 0x400
        or bool(attributes[0] & 0x10) != directory
    ):
        api.CloseHandle(handle)
        raise _Unsafe("unsafe_reparse_or_type")
    return handle


@dataclass(frozen=True)
class _WinDirectory:
    handle: int
    path: Path


class _WindowsFS:
    """NTFS contract: retained no-share-delete ancestors, handle rename/delete.

    Private staging requires an owner/System protected DACL. Unsupported APIs,
    ACLs, reparse points or sharing modes fail closed, without os.rename/unlink.
    Native OS/volume acceptance is separate from portable adapter tests.
    """

    def __init__(self, *, api=None, handle_for_fd=None):
        self.api = api or _windows_api()
        if handle_for_fd is None:
            import msvcrt

            handle_for_fd = msvcrt.get_osfhandle
        self.handle_for_fd = handle_for_fd
        if api is None:
            self.api.SetFileInformationByHandle.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            self.api.SetFileInformationByHandle.restype = ctypes.c_int

    @contextmanager
    def directory(self, path):
        with ExitStack() as stack:
            for part in (*reversed(path.parents), path):
                handle = _windows_existing(self.api, part, directory=True)
                stack.callback(self.api.CloseHandle, handle)
            yield _WinDirectory(handle, path)

    def names(self, directory, path, *, limit=_MAX_OBJECTS):
        names = set()
        with os.scandir(directory.path) as entries:
            for entry in entries:
                names.add(entry.name)
                if len(names) > limit:
                    raise _Unsafe("directory_limit")
        return names

    @contextmanager
    def file(self, directory, path, *, mutable=False):
        import msvcrt

        handle = _windows_existing(self.api, path, directory=False, mutable=mutable)
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            self.api.CloseHandle(handle)
            raise
        try:
            _identity_of(os.fstat(descriptor))
            yield descriptor
        finally:
            os.close(descriptor)  # Owns and closes the native handle.

    def device(self, directory):
        # Ancestors cannot be renamed while the directory handle is retained.
        return os.stat(directory.path, follow_symlinks=False).st_dev

    def mkdir(self, directory, path):
        _windows_private_directory(path, create=True)

    def private(self, directory):
        _windows_private_directory(directory.path, create=False)

    def rename(self, source, source_path, target, target_path, descriptor):
        # FILE_RENAME_INFO; ReplaceIfExists=False, an absolute destination below
        # retained ancestor handles. Never substitute an overwrite/path rename.
        encoded = str(target_path).encode("utf-16-le")

        class RenameInfo(ctypes.Structure):
            _fields_ = [
                ("ReplaceIfExists", ctypes.c_ubyte),
                ("RootDirectory", ctypes.c_void_p),
                ("FileNameLength", ctypes.c_uint32),
                ("FileName", ctypes.c_ubyte * len(encoded)),
            ]

        value = RenameInfo()
        value.FileNameLength = len(encoded)
        value.FileName[:] = encoded
        self._set(descriptor, 3, value)

    def unlink(self, directory, path, descriptor):
        # FILE_DISPOSITION_INFO on the verified open handle. Last close performs
        # the physical deletion; the caller checkpoints success after that close.
        self._set(descriptor, 4, ctypes.c_ubyte(1))

    def _set(self, descriptor, kind, value):
        if not self.api.SetFileInformationByHandle(
            self.handle_for_fd(descriptor), kind, ctypes.byref(value), ctypes.sizeof(value)
        ):
            raise OSError(self.api.GetLastError(), "media_handle_mutation_failed")


def _windows_private_directory(path, *, create):
    """Create/verify a protected owner/System-only staging DACL; no ACL fallback."""
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    advapi.GetNamedSecurityInfoW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_wchar_p),
        ctypes.c_void_p,
    ]
    kernel.CreateDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    expected = "D:P(A;;FA;;;SY)(A;;FA;;;OW)"
    if create:
        descriptor = ctypes.c_void_p()
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            expected, 1, ctypes.byref(descriptor), None
        ):
            raise _Unsafe("private_acl_unavailable")

        class SecurityAttributes(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("descriptor", ctypes.c_void_p),
                ("inherit", ctypes.c_int),
            ]

        attrs = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        try:
            if (
                not kernel.CreateDirectoryW(str(path), ctypes.byref(attrs))
                and ctypes.get_last_error() != 183
            ):
                raise _Unsafe("private_directory_unavailable")
        finally:
            kernel.LocalFree(descriptor)
    descriptor = ctypes.c_void_p()
    text = ctypes.c_wchar_p()
    if advapi.GetNamedSecurityInfoW(
        str(path), 1, 4, None, None, None, None, ctypes.byref(descriptor)
    ):
        raise _Unsafe("private_acl_unavailable")
    try:
        if not advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor, 1, 4, ctypes.byref(text), None
        ):
            raise _Unsafe("private_acl_unavailable")
        if text.value != expected:
            raise _Unsafe("unsafe_staging_permissions")
    finally:
        if text:
            kernel.LocalFree(text)
        kernel.LocalFree(descriptor)


def _port(platform):
    if platform == "posix":
        return _PosixFS()
    if platform == "nt":
        try:
            return _WindowsFS()
        except (AttributeError, OSError):
            raise _Unsafe("windows_native_api_unavailable") from None
    raise _Unsafe("unsupported_platform")


def _read_json(fs, path):
    with fs.directory(path.parent) as parent, fs.file(parent, path) as descriptor:
        if os.fstat(descriptor).st_size > _MAX_JSON:
            raise _Unsafe("metadata_limit")
        data = os.read(descriptor, _MAX_JSON + 1)
        if len(data) > _MAX_JSON:
            raise _Unsafe("metadata_limit")
        return _json(data.decode())


def _hash(descriptor, cancelled):
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        if cancelled():
            raise InterruptedError("cleanup_cancelled")
        chunk = os.read(descriptor, _CHUNK)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _verify(descriptor, obj, cancelled):
    before = _identity_of(os.fstat(descriptor))
    if (
        before != obj["identity"]
        or before[2] != obj["size"]
        or _hash(descriptor, cancelled) != obj["sha256"]
        or _identity_of(os.fstat(descriptor)) != before
    ):
        raise _Unsafe("changed_object")


def _source_manifest(manifest, payload):
    if any(key in manifest for key in ("storage_kind", "segment_root", "recording_root")):
        raise _Unsafe("shared_source_layout")
    for key in (
        "schema_version",
        "source_epoch_id",
        "start_frame_id",
        "end_frame_id",
        "media_sha256",
    ):
        if manifest.get(key) != payload.get(key):
            raise _Unsafe("source_lineage")
    segments = manifest.get("segment_sha256")
    if not isinstance(segments, dict) or not 1 <= len(segments) < _MAX_OBJECTS:
        raise _Unsafe("source_inventory")
    for name, digest in segments.items():
        if not _name(name).endswith((".mjpeg", ".mjpgseg")) or not _sha(digest):
            raise _Unsafe("source_inventory")
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "source_epoch_id",
            "width",
            "height",
            "start_frame_id",
            "end_frame_id",
            "frame_count",
        )
    }
    if (
        any(
            type(identity[key]) is not int or identity[key] <= 0
            for key in ("width", "height", "frame_count")
        )
        or identity["frame_count"] != identity["end_frame_id"] - identity["start_frame_id"] + 1
    ):
        raise _Unsafe("source_inventory")
    identity["segments"] = [{"name": name, "sha256": digest} for name, digest in segments.items()]
    if (
        hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != payload["media_sha256"]
    ):
        raise _Unsafe("source_digest")
    return segments


def _load_job(fs, jobs_db, claim, *, require_artifact=True, include_state=False):
    with fs.directory(jobs_db.parent) as parent, fs.file(parent, jobs_db):
        # No job-store construction/migration. Read committed WAL data as well.
        connection = sqlite3.connect(jobs_db.as_uri() + "?mode=ro", uri=True, timeout=1)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            row, payload = _job(connection, claim.job_id, claim.session_id, claim.shot_id)
            if require_artifact and (
                row["status"] != "available" or row["artifact_id"] != claim.artifact_id
            ):
                raise _Unsafe("job_unresolved")
            return (row["status"], row["artifact_id"], payload) if include_state else payload
        finally:
            connection.close()


def _typed_media_specs(fs, jobs_db, claim, root, source_payload, progress=None):
    """Inventory validated sibling playback output for this original shot only."""

    source_encoded = json.dumps(
        source_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    expected = ShotMediaEncodeJobV1.create(
        session_id=claim.session_id,
        shot_id=claim.shot_id,
        source_epoch_id=source_payload["source_epoch_id"],
        start_frame_id=source_payload["start_frame_id"],
        end_frame_id=source_payload["end_frame_id"],
        media_root=Path(source_payload["media_root"]),
        media_sha256=source_payload["media_sha256"],
        source=LegacyFrozenJobMediaSourceV1(
            kind="legacy_frozen_job",
            postgame_manifest_sha256=claim.postgame_manifest_sha256,
            source_job_id=claim.job_id,
            source_job_payload_sha256=hashlib.sha256(source_encoded.encode()).hexdigest(),
        ),
    )
    with fs.directory(jobs_db.parent) as parent, fs.file(parent, jobs_db):
        connection = sqlite3.connect(jobs_db.as_uri() + "?mode=ro", uri=True, timeout=1)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(shot_analysis_jobs)")
            }
            if "job_kind" not in columns:
                return [], {}
            rows = connection.execute(
                "SELECT payload_json,payload_sha256,status,artifact_id "
                "FROM shot_analysis_jobs WHERE job_kind='media' AND job_id=?",
                (expected.job_id,),
            ).fetchall()
        finally:
            connection.close()
    specs = []
    directories = {}
    if len(rows) > 1:
        raise _Unsafe("ambiguous_media_task_metadata")
    for row in rows:
        encoded = row["payload_json"]
        if (
            not isinstance(encoded, str)
            or hashlib.sha256(encoded.encode()).hexdigest() != row["payload_sha256"]
        ):
            raise _Unsafe("invalid_media_task_metadata")
        try:
            job = _media_job_from_payload(json.loads(encoded))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise _Unsafe("invalid_media_task_metadata") from exc
        # max_attempts is retry policy, not content identity; all other fields
        # must still be the exact deterministic sibling task.
        left, right = job.identity(), expected.identity()
        left.pop("max_attempts")
        right.pop("max_attempts")
        if job.job_id != expected.job_id or left != right:
            raise _Unsafe("invalid_media_task_lineage")
        parent = root / "postgame" / claim.session_id
        prefix = f".pending-recording-playback-{expected.job_id}-"
        retained = {}
        if progress:
            objects = progress.get("objects")
            if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
                raise _Unsafe("invalid_journal")
            for obj in objects:
                path = Path(obj.get("path", ""))
                if path.parent.parent == parent and path.parent.name.startswith(prefix):
                    if obj.get("root") != "artifacts" or path.name not in {
                        "manifest.json",
                        "original.mp4",
                    }:
                        raise _Unsafe("invalid_media_temporary_inventory")
                    retained.setdefault(path.parent.name, {})[path.name] = obj
        try:
            with fs.directory(parent) as artifact_parent:
                parent_names = fs.names(artifact_parent, parent, limit=_MAX_ARTIFACT_ENTRIES)
        except FileNotFoundError:
            # Scheduling precedes the first encoder mkdir. With no published
            # output yet, frozen copied media still has deletion authority.
            parent_names = set()
        observed_pending = {name for name in parent_names if name.startswith(prefix)}
        if progress:
            for name in observed_pending - retained.keys():
                # A crash before the writer opens leaves an empty directory,
                # which has no deletion object to journal. Never adopt new files.
                with fs.directory(parent / name) as empty_directory:
                    if fs.names(empty_directory, parent / name):
                        raise _Unsafe("new_media_temporary_inventory")
                retained[name] = {}
        pending_names = sorted(observed_pending | retained.keys())
        for name in pending_names:
            suffix = name.removeprefix(prefix)
            if len(suffix) != 32 or any(
                character not in "0123456789abcdef" for character in suffix
            ):
                raise _Unsafe("invalid_media_temporary_identity")
            pending = parent / name
            with fs.directory(pending) as pending_directory:
                children = fs.names(pending_directory, pending)
                if not children <= {"manifest.json", "original.mp4"}:
                    raise _Unsafe("invalid_media_temporary_inventory")
                expected_children = set(retained[name]) if progress else children
                if children - expected_children:
                    raise _Unsafe("new_media_temporary_inventory")
                for child in sorted(expected_children):
                    path = pending / child
                    if progress:
                        obj = retained[name][child]
                        specs.append(("artifacts", path, obj.get("sha256"), obj.get("size")))
                        continue
                    with fs.file(pending_directory, path) as descriptor:
                        identity = _identity_of(os.fstat(descriptor))
                        if not 0 <= identity[2] <= _MAX_MEDIA_BYTES:
                            raise _Unsafe("media_size_limit")
                        digest = _hash(descriptor, lambda: False)
                    specs.append(("artifacts", path, digest, identity[2]))
            directories[pending] = expected_children
        if row["status"] != "available" and row["artifact_id"] is not None:
            raise _Unsafe("unresolved_media_task_artifact")
        # Publishing precedes the job transaction. A committed directory is
        # authoritative through its typed manifest even if completion was lost.
        published = sorted(name for name in parent_names if name.startswith("recording-playback-"))
        found = set()
        for artifact_id in published:
            if not _sha(artifact_id.removeprefix("recording-playback-")):
                raise _Unsafe("invalid_media_task_artifact")
            output = parent / artifact_id
            metadata = _read_json(fs, output / "manifest.json")
            identity = metadata.get("job")
            if not isinstance(identity, dict):
                raise _Unsafe("invalid_media_task_artifact")
            if identity.get("job_id") != job.job_id:
                continue
            try:
                artifact = recording_playback_inventory(output, job)
            except RecordingPlaybackError as exc:
                raise _Unsafe("invalid_media_task_artifact") from exc
            specs.append(
                ("artifacts", artifact.media_path, artifact.content_sha256, artifact.byte_size)
            )
            directories[output] = {"manifest.json", "original.mp4"}
            found.add(artifact_id)
        if row["status"] == "available" and row["artifact_id"] not in found:
            raise _Unsafe("invalid_media_task_artifact")
    return specs, directories


def _canonical_media_specs(fs, claim, root, payload, progress=None, *, existing_artifact_id=None):
    """Find committed canonical siblings, including publish-before-commit crashes."""
    from billiards_ai.lab.mainline1_shot_local_video import (
        _artifact_identity_payload,
        _canonical_bytes,
    )

    parent = root / "postgame" / claim.session_id
    try:
        with fs.directory(parent) as directory:
            names = fs.names(directory, parent, limit=_MAX_ARTIFACT_ENTRIES)
    except FileNotFoundError:
        return [], {}
    expected = {
        key: payload[key]
        for key in (
            "schema_version",
            "session_id",
            "job_id",
            "source_epoch_id",
            "media_sha256",
            "capture_setup_id",
            "calibration_artifact_id",
            "config_sha256",
            "model_package_id",
            "identity_training_package_id",
            "identity_training_package_sha256",
        )
    }
    expected.update(
        kind="mainline1_shot_local_video",
        artifact_role="usb_postgame_shot",
        source_kind="usb_live",
        live_shot_id=claim.shot_id,
    )
    prefix = ".pending-shot-" + hashlib.sha256(_canonical_bytes(expected)).hexdigest() + "-"
    retained = {}
    if progress is not None and not isinstance(progress, dict):
        raise _Unsafe("invalid_journal")
    if progress:
        objects = progress.get("objects")
        if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
            raise _Unsafe("invalid_journal")
        for obj in objects:
            path = Path(obj.get("path", ""))
            if path.parent.parent == parent and path.parent.name.startswith(prefix):
                if obj.get("root") != "artifacts" or path.name != "overlay.mp4":
                    raise _Unsafe("invalid_canonical_temporary_inventory")
                retained[path.parent.name] = obj
    specs, directories = [], {}
    for name in sorted(names | retained.keys()):
        if name.startswith(".pending-") and len(name) == len(".pending-") + 32:
            # Historic staging names lack a source binding. Never infer which
            # shot owns those bytes or report them successfully reclaimed.
            raise _Unsafe("unattributed_canonical_temporary")
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            if len(suffix) != 32 or any(c not in "0123456789abcdef" for c in suffix):
                raise _Unsafe("invalid_canonical_temporary_identity")
            pending = parent / name
            with fs.directory(pending) as directory:
                children = fs.names(directory, pending)
                if children - (_FACT_NAMES | {"manifest.json", "overlay.mp4"}):
                    raise _Unsafe("invalid_canonical_temporary_inventory")
                media = pending / "overlay.mp4"
                if name in retained:
                    obj = retained[name]
                    specs.append(("artifacts", media, obj.get("sha256"), obj.get("size")))
                    children.add("overlay.mp4")
                elif "overlay.mp4" in children:
                    if progress:
                        raise _Unsafe("new_canonical_temporary_inventory")
                    with fs.file(directory, media) as descriptor:
                        size = _identity_of(os.fstat(descriptor))[2]
                        if size > _MAX_MEDIA_BYTES:
                            raise _Unsafe("media_size_limit")
                        specs.append(("artifacts", media, _hash(descriptor, lambda: False), size))
            directories[pending] = children
    for name in sorted(names):
        if not _sha(name) or name == existing_artifact_id:
            continue
        output = parent / name
        manifest = _read_json(fs, output / "manifest.json")
        if manifest.get("job_id") != claim.job_id:
            continue
        published_identity = expected | {
            "artifact_id": name,
            "artifact_id_algorithm": "identity-semantic-inventory-v2",
        }
        inventory = manifest.get("inventory")
        if (
            any(manifest.get(key) != value for key, value in published_identity.items())
            or not isinstance(inventory, dict)
            or set(inventory) != _FACT_NAMES | {"overlay.mp4"}
            or hashlib.sha256(_canonical_bytes(_artifact_identity_payload(manifest))).hexdigest()
            != name
        ):
            raise _Unsafe("canonical_sibling_lineage")
        video = inventory["overlay.mp4"]
        if (
            not isinstance(video, dict)
            or set(video) != {"sha256", "bytes"}
            or not _sha(video["sha256"])
            or type(video["bytes"]) is not int
            or not 0 < video["bytes"] <= _MAX_MEDIA_BYTES
        ):
            raise _Unsafe("canonical_sibling_inventory")
        specs.append(("artifacts", output / "overlay.mp4", video["sha256"], video["bytes"]))
        directories[output] = {"manifest.json", *inventory}
    return specs, directories


def _specs(fs, claim, jobs_db, root, jobs, *, validate_payload=lambda payload: None, progress=None):
    for value in (claim.session_id, claim.shot_id, claim.job_id):
        if _identity(value) in {".", ".."}:
            raise _Unsafe("unsafe_identity")
    if (
        not _sha(claim.artifact_id)
        or not _sha(claim.content_sha256)
        or not _sha(claim.postgame_manifest_sha256)
        or claim.content_type != "video/mp4"
        or type(claim.byte_size) is not int
        or claim.byte_size <= 0
        or claim.clip_id
        != session_scoped_clip_id(claim.session_id, claim.shot_id, claim.job_id, claim.artifact_id)
    ):
        raise _Unsafe("discard_identity")
    if claim.local_path.name == "original.mp4":
        expected = (
            root
            / "postgame"
            / claim.session_id
            / ("recording-playback-" + claim.artifact_id)
            / "original.mp4"
        )
        if claim.local_path != expected:
            raise _Unsafe("original_playback_path")
        status, canonical_id, payload = _load_job(
            fs,
            jobs_db,
            claim,
            require_artifact=False,
            include_state=True,
        )
        validate_payload(payload)
        raw = jobs / "media" / claim.session_id / claim.job_id
        if payload["media_root"] != str(raw):
            raise _Unsafe("unsupported_source_layout")
        interval = _read_json(fs, raw / "interval.json")
        segments = _source_manifest(interval, payload)
        specs, directories = _typed_media_specs(
            fs,
            jobs_db,
            claim,
            root,
            payload,
            progress,
        )
        if ("artifacts", claim.local_path, claim.content_sha256, claim.byte_size) not in specs:
            raise _Unsafe("selected_original_playback_identity")
        canonical_specs, canonical_directories = _canonical_media_specs(
            fs, claim, root, payload, progress
        )
        specs += canonical_specs
        directories.update(canonical_directories)
        if status == "available" and (
            canonical_id is None
            or root / "postgame" / claim.session_id / canonical_id not in canonical_directories
        ):
            raise _Unsafe("canonical_sibling_identity")
        specs += [("jobs", raw / name, digest, None) for name, digest in sorted(segments.items())]
        directories[raw] = {"interval.json", *segments}
        return specs, directories
    output = root / "postgame" / claim.session_id / claim.artifact_id
    if claim.local_path != output / "overlay.mp4":
        raise _Unsafe("overlay_path")
    payload = _load_job(fs, jobs_db, claim)
    validate_payload(payload)
    raw = jobs / "media" / claim.session_id / claim.job_id
    if payload["media_root"] != str(raw):
        raise _Unsafe("unsupported_source_layout")
    interval = _read_json(fs, raw / "interval.json")
    segments = _source_manifest(interval, payload)
    manifest = _read_json(fs, output / "manifest.json")
    expected = {
        key: payload[key]
        for key in (
            "schema_version",
            "session_id",
            "job_id",
            "source_epoch_id",
            "media_sha256",
            "capture_setup_id",
            "calibration_artifact_id",
            "config_sha256",
            "model_package_id",
            "identity_training_package_id",
            "identity_training_package_sha256",
        )
    }
    expected.update(
        kind="mainline1_shot_local_video",
        artifact_role="usb_postgame_shot",
        source_kind="usb_live",
        live_shot_id=claim.shot_id,
        artifact_id=claim.artifact_id,
    )
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise _Unsafe("overlay_lineage")
    inventory = manifest.get("inventory")
    if (
        not isinstance(inventory, dict)
        or "overlay.mp4" not in inventory
        or not set(inventory) <= _FACT_NAMES | {"overlay.mp4"}
        or inventory["overlay.mp4"] != {"bytes": claim.byte_size, "sha256": claim.content_sha256}
    ):
        raise _Unsafe("overlay_inventory")
    specs = [("artifacts", claim.local_path, claim.content_sha256, claim.byte_size)]
    specs += [("jobs", raw / name, digest, None) for name, digest in sorted(segments.items())]
    media_specs, media_directories = _typed_media_specs(fs, jobs_db, claim, root, payload, progress)
    specs += media_specs
    canonical_specs, canonical_directories = _canonical_media_specs(
        fs,
        claim,
        root,
        payload,
        progress,
        existing_artifact_id=claim.artifact_id,
    )
    specs += canonical_specs
    media_directories.update(canonical_directories)
    directories = {raw: {"interval.json", *segments}, output: {"manifest.json", *inventory}}
    for path, names in media_directories.items():
        if path in directories:
            raise _Unsafe("media_artifact_path_conflict")
        directories[path] = names
    return specs, directories


def _stage_path(root, claim, index, namespace=".local-media-cleanup"):
    key = hashlib.sha256(claim.clip_id.encode()).hexdigest()
    return root / namespace / key / f"{index:03d}.detached"


def _check_names(fs, directories, progress=None):
    removed = set()
    if progress:
        removed = {Path(obj["path"]) for obj in progress["objects"] if obj["state"] != "planned"}
    for path, expected in directories.items():
        with fs.directory(path) as directory:
            actual = fs.names(directory, path)
            # During recovery planned objects may already have been detached.
            allowed_missing = {obj.name for obj in removed if obj.parent == path}
            if progress:
                allowed_missing |= {
                    Path(obj["path"]).name
                    for obj in progress["objects"]
                    if Path(obj["path"]).parent == path
                }
            if actual - expected or expected - actual - allowed_missing:
                raise _Unsafe("unknown_or_missing_children")
            for name in actual:
                with fs.file(directory, path / name):
                    pass


def _inventory(fs, specs, roots, claim, cancelled, *, namespace=".local-media-cleanup"):
    objects = []
    for index, (kind, path, digest, expected_size) in enumerate(specs):
        with fs.directory(path.parent) as parent, fs.file(parent, path) as descriptor:
            identity = _identity_of(os.fstat(descriptor))
            if not 0 <= identity[2] <= _MAX_MEDIA_BYTES:
                raise _Unsafe("media_size_limit")
            obj = dict(
                path=str(path),
                root=kind,
                staged_path=str(_stage_path(roots[kind], claim, index, namespace)),
                sha256=digest,
                size=identity[2] if expected_size is None else expected_size,
                identity=identity,
                state="planned",
                reclaimed_bytes=0,
            )
            _verify(descriptor, obj, cancelled)
            objects.append(obj)
    return dict(version=1, clip_id=claim.clip_id, objects=objects)


def _validate_progress(progress, specs, roots, claim, *, namespace=".local-media-cleanup"):
    if (
        progress.get("version") != 1
        or progress.get("clip_id") != claim.clip_id
        or not isinstance(progress.get("objects"), list)
        or len(progress["objects"]) != len(specs)
    ):
        raise _Unsafe("invalid_journal")
    for index, (obj, (kind, path, digest, size)) in enumerate(
        zip(progress["objects"], specs, strict=True)
    ):
        if (
            not isinstance(obj, dict)
            or obj.get("path") != str(path)
            or obj.get("root") != kind
            or obj.get("staged_path") != str(_stage_path(roots[kind], claim, index, namespace))
            or obj.get("sha256") != digest
            or type(obj.get("size")) is not int
            or not 0 <= obj["size"] <= _MAX_MEDIA_BYTES
            or size is not None
            and size != obj["size"]
            or not isinstance(obj.get("identity"), list)
            or len(obj["identity"]) != 4
            or any(type(value) is not int for value in obj["identity"])
            or obj["identity"][2] != obj["size"]
            or not obj["identity"][1]
            or obj.get("state") not in {"planned", "detached", "unlinking", "deleted", "absent"}
            or type(obj.get("reclaimed_bytes")) is not int
            or obj["reclaimed_bytes"] not in {0, obj["size"]}
            or obj["state"] != "deleted"
            and obj["reclaimed_bytes"]
        ):
            raise _Unsafe("invalid_journal")


def _checkpoint(store, claim, progress):
    if not store.update_local_discard_progress(claim, _canonical(progress)):
        raise _Stale()


def _exists(fs, directory, path):
    try:
        with fs.file(directory, path):
            return True
    except FileNotFoundError:
        return False


@contextmanager
def _disposal_handles(fs, progress, roots, obj, checkpoint):
    source, staged = Path(obj["path"]), Path(obj["staged_path"])
    root = roots[obj["root"]]
    with ExitStack() as stack:
        parent = stack.enter_context(fs.directory(root))
        for path in (staged.parent.parent, staged.parent):
            checkpoint(progress)
            fs.mkdir(parent, path)
            parent = stack.enter_context(fs.directory(path))
            fs.private(parent)
        stage = parent
        source_parent = stack.enter_context(fs.directory(source.parent))
        if fs.device(source_parent) != fs.device(stage):
            raise _Unsafe("cross_device_staging")
        expected = {
            Path(item["staged_path"]).name
            for item in progress["objects"]
            if item["root"] == obj["root"]
        }
        if fs.names(stage, staged.parent) - expected:
            raise _Unsafe("unknown_staging_children")
        yield source_parent, source, stage, staged


def _object_location(fs, obj, source_parent, source, stage, staged):
    in_source, in_stage = _exists(fs, source_parent, source), _exists(fs, stage, staged)
    if in_source and in_stage:
        raise _Unsafe("duplicate_or_replacement_object")
    if obj["state"] in {"deleted", "absent"}:
        if in_source or in_stage:
            raise _Unsafe("reappeared_object")
    elif not in_source and not in_stage:
        if obj["state"] != "unlinking":
            raise _Unsafe("missing_inventoried_object")
    elif in_source and obj["state"] != "planned":
        raise _Unsafe("replacement_object")
    return in_source, in_stage


def _preflight_disposal(fs, progress, roots, cancelled, checkpoint):
    """Check every source/staged object before owner mutation and completion."""
    for obj in progress["objects"]:
        if cancelled():
            raise InterruptedError("cleanup_cancelled")
        with _disposal_handles(fs, progress, roots, obj, checkpoint) as handles:
            source_parent, source, stage, staged = handles
            in_source, in_stage = _object_location(fs, obj, *handles)
            if in_source or in_stage:
                parent, path = (source_parent, source) if in_source else (stage, staged)
                with fs.file(parent, path) as descriptor:
                    _verify(descriptor, obj, cancelled)


def _dispose(fs, progress, roots, cancelled, fault_hook, checkpoint):
    for obj in progress["objects"]:
        if cancelled():
            raise InterruptedError("cleanup_cancelled")
        with _disposal_handles(fs, progress, roots, obj, checkpoint) as handles:
            source_parent, source, stage, staged = handles
            in_source, in_stage = _object_location(fs, obj, *handles)
            if obj["state"] in {"deleted", "absent"}:
                continue
            if not in_source and not in_stage:
                obj["state"] = "absent"
                checkpoint(progress)
                continue
            if in_source:
                with fs.file(source_parent, source, mutable=True) as descriptor:
                    _verify(descriptor, obj, cancelled)
                    checkpoint(progress)
                    fs.rename(source_parent, source, stage, staged, descriptor)
                fault_hook("after_detach", obj)
            with fs.file(stage, staged, mutable=True) as descriptor:
                _verify(descriptor, obj, cancelled)
                obj["state"] = "detached"
                checkpoint(progress)
                obj["state"] = "unlinking"
                checkpoint(progress)
                _verify(descriptor, obj, cancelled)
                checkpoint(progress)
                fs.unlink(stage, staged, descriptor)
            fault_hook("after_unlink", obj)
            obj["state"], obj["reclaimed_bytes"] = "deleted", obj["size"]
            checkpoint(progress)


def process_local_discard_once(
    store,
    jobs_db: Path,
    artifact_root: Path,
    jobs_root: Path,
    *,
    worker_id="local-media-cleanup",
    now_utc=None,
    lease_seconds=30,
    cancelled=lambda: False,
    fault_hook=lambda point, obj: None,
    platform=None,
    session_ids=None,
    exclude_managed=False,
) -> LocalDiscardAttempt:
    """Serialize claim/takeover under a persistent database control pin, then dispose.

    The control pin lives in a distinct namespace from real session pins and is
    held from before claiming through the last mutation/terminal checkpoint. All
    collector processes must use this entrypoint: a lease may expire during long
    hashes, but another collector cannot replace its token until this owner exits.
    The session pin remains authoritative for reader/writer media exclusion.
    """
    with ExitStack() as stack:
        try:
            if cancelled():
                raise InterruptedError("cleanup_cancelled")
            root = _absolute(artifact_root)
            _assert_directory_chain(root)
            store_path = _absolute(store.path)
            _assert_directory_chain(store_path.parent)
            control = store_path.parent / ".local-media-cleanup-control"
            control.mkdir(mode=0o700, exist_ok=True)
            # Scope claims to the canonical DB, independently of caller media roots.
            # This persistent namespace cannot alias a real session's media pin.
            claim_identity = "claims-v1:" + hashlib.sha256(str(store_path).encode()).hexdigest()
            stack.enter_context(session_media_lock(control, claim_identity, exclusive=True))
        except (OSError, ValueError) as exc:
            retry = isinstance(exc, (TimeoutError, InterruptedError, PermissionError))
            return LocalDiscardAttempt(
                "retry" if retry else "blocked", "collector_control_unavailable"
            )
        return _process_local_discard_under_claim_pin(
            store,
            jobs_db,
            root,
            jobs_root,
            worker_id=worker_id,
            now_utc=now_utc,
            lease_seconds=lease_seconds,
            cancelled=cancelled,
            fault_hook=fault_hook,
            platform=platform,
            session_ids=session_ids,
            exclude_managed=exclude_managed,
        )


def _process_local_discard_under_claim_pin(
    store,
    jobs_db,
    artifact_root,
    jobs_root,
    *,
    worker_id,
    now_utc,
    lease_seconds,
    cancelled,
    fault_hook,
    platform,
    session_ids=None,
    exclude_managed=False,
):
    """Consume one explicit immutable authority. No scans, TTL, or job-store init.

    artifact_root is the configured capture root (also the session pin root),
    not the per-session postgame output directory. Work is chunk-cancellable.
    ``reclaimed_bytes`` counts confirmed successful unlinks in the durable journal;
    crash-window missing objects are recorded absent with zero claimed bytes.
    """
    now = now_utc or datetime.now(UTC)
    selection = {} if session_ids is None else {"session_ids": session_ids}
    if exclude_managed:
        selection["exclude_managed"] = True
    claim = store.claim_due_local_discard(
        worker_id, now, lease_seconds=lease_seconds, artifact_root=artifact_root, **selection
    )
    if claim is None:
        return LocalDiscardAttempt("idle")
    progress = None
    try:
        root, jobs, db = map(_absolute, (artifact_root, jobs_root, jobs_db))
        if store.artifact_path_for_session(claim.session_id) != root:
            raise _Unsafe("pin_root_mismatch")
        fs = _port(platform or os.name)
        with session_media_lock(root, claim.session_id, exclusive=True):
            # Fence before reading or changing recovery state, including on restart.
            if not store.update_local_discard_progress(claim, claim.progress_json):
                raise _Stale()
            loaded = _json(claim.progress_json)
            specs, directories = _specs(fs, claim, db, root, jobs, progress=loaded)
            roots = {"artifacts": root, "jobs": jobs}
            if loaded:
                _validate_progress(loaded, specs, roots, claim)
            progress = loaded
            _check_names(fs, directories, progress)
            if not progress:
                progress = _inventory(fs, specs, roots, claim, cancelled)
            _checkpoint(store, claim, progress)
            _dispose(
                fs,
                progress,
                roots,
                cancelled,
                fault_hook,
                lambda value: _checkpoint(store, claim, value),
            )
            if not store.complete_local_discard(claim, progress_json=_canonical(progress)):
                raise _Stale()
        return LocalDiscardAttempt(
            "completed",
            reclaimed_bytes=sum(obj["reclaimed_bytes"] for obj in progress["objects"]),
            session_id=claim.session_id,
            shot_id=claim.shot_id,
        )
    except _Stale:
        return LocalDiscardAttempt(
            "stale", "claim_lost", session_id=claim.session_id, shot_id=claim.shot_id
        )
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        # Busy pins, resource contention, cancellation and I/O interruptions retry.
        retry = (
            isinstance(exc, (TimeoutError, InterruptedError, PermissionError))
            or isinstance(exc, OSError)
            and exc.errno in {errno.EBUSY, errno.EAGAIN, errno.EINTR, errno.ENOSPC, errno.EIO}
        )
        code = (
            str(exc)
            if isinstance(exc, _Unsafe)
            else ("media_busy" if retry else "unsafe_or_invalid_media")
        )
        try:
            encoded = None if progress is None else _canonical(progress)
        except _Unsafe:
            # An oversized inventory was never accepted or used for a mutation.
            # Preserve the previous durable checkpoint instead of retrying forever.
            encoded = None
        if retry:
            accepted = store.retry_local_discard(
                claim,
                code,
                now + timedelta(seconds=min(2 ** min(claim.attempts, 8), 300)),
                progress_json=encoded,
            )
        else:
            accepted = store.block_local_discard(claim, code, progress_json=encoded)
        return (
            LocalDiscardAttempt(
                "retry" if retry else "blocked",
                code,
                sum(obj.get("reclaimed_bytes", 0) for obj in (progress or {}).get("objects", [])),
                claim.session_id,
                claim.shot_id,
            )
            if accepted
            else LocalDiscardAttempt("stale", "claim_lost")
        )


def _owner_retryable(exc):
    # Store authority revalidation preserves SQLite errors as a ValueError cause.
    if isinstance(exc, ValueError) and isinstance(exc.__cause__, sqlite3.Error):
        exc = exc.__cause__
    return (
        isinstance(exc, (TimeoutError, InterruptedError, PermissionError))
        or isinstance(exc, OSError)
        and exc.errno in {errno.EBUSY, errno.EAGAIN, errno.EINTR, errno.ENOSPC, errno.EIO}
        or isinstance(exc, sqlite3.Error)
        and getattr(exc, "sqlite_errorcode", -1) & 0xFF
        in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
            sqlite3.SQLITE_IOERR,
            sqlite3.SQLITE_FULL,
            sqlite3.SQLITE_INTERRUPT,
        }
    )


def _owner_source_identity(store, claim, scope, record, payload):
    media = store.get_live_shot_media(claim.session_id, claim.shot_id)
    epoch = store.get_managed_capture_epoch(scope, claim.session_id, payload["source_epoch_id"])
    if (
        epoch.spec_sha256 != record.spec_sha256
        or epoch.open_request_id != record.open_request_id
        or media is None
        or (media.job_id, media.source_epoch_id, media.media_sha256, media.media_path)
        != (
            claim.job_id,
            payload["source_epoch_id"],
            payload["media_sha256"],
            payload["media_root"],
        )
    ):
        raise _Unsafe("original_source_mismatch")
    expected = {
        "capture_setup_id": record.spec.capture_setup_id,
        "calibration_artifact_id": record.spec.calibration_artifact_id,
        "config_sha256": record.spec.config_sha256,
        "model_package_id": record.spec.model_package_id,
        "identity_training_package_id": record.spec.identity_training_package_id,
        "identity_training_package_sha256": record.spec.identity_training_package_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise _Unsafe("original_job_dependencies_mismatch")


def process_local_owner_delete_once(
    store,
    jobs_db: Path,
    artifact_root: Path,
    jobs_root: Path,
    *,
    worker_id="local-owner-delete",
    now_utc=None,
    lease_seconds=30,
    cancelled=lambda: False,
    fault_hook=lambda point, obj: None,
    platform=None,
    session_ids=None,
) -> LocalOwnerDeleteAttempt:
    """Erase one accepted owner task under independent control and session pins.

    The caller resolves cleanup resources. This executor independently compares
    every supplied root with the accepted task's retained original resource spec;
    missing evidence never falls back to the current runtime. Only the owner
    ledger provides checkpoints and completion authority for these saved copies.
    The canonical store path anchors preclaim exclusion independently of supplied
    media roots, so an invalid-root caller cannot rotate an active local token.
    """
    with ExitStack() as stack:
        try:
            if cancelled():
                raise InterruptedError("cleanup_cancelled")
            root = _absolute(artifact_root)
            _assert_directory_chain(root)
            store_path = _absolute(store.path)
            _assert_directory_chain(store_path.parent)
            control = store_path.parent / ".local-owner-media-cleanup-control"
            control.mkdir(mode=0o700, exist_ok=True)
            claim_identity = "claims-v1:" + hashlib.sha256(str(store_path).encode()).hexdigest()
            stack.enter_context(session_media_lock(control, claim_identity, exclusive=True))
        except (OSError, ValueError) as exc:
            return LocalOwnerDeleteAttempt(
                "retry" if _owner_retryable(exc) else "blocked", "collector_control_unavailable"
            )
        return _process_local_owner_under_claim_pin(
            store,
            jobs_db,
            root,
            jobs_root,
            worker_id=worker_id,
            now_utc=now_utc,
            lease_seconds=lease_seconds,
            cancelled=cancelled,
            fault_hook=fault_hook,
            platform=platform,
            session_ids=session_ids,
        )


def _process_local_owner_under_claim_pin(
    store,
    jobs_db,
    artifact_root,
    jobs_root,
    *,
    worker_id,
    now_utc,
    lease_seconds,
    cancelled,
    fault_hook,
    platform,
    session_ids,
):
    now = now_utc or datetime.now(UTC)
    try:
        claim = store.claim_due_owner_delete(
            worker_id, now, lease_seconds=lease_seconds, session_ids=session_ids
        )
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        return LocalOwnerDeleteAttempt(
            "retry" if _owner_retryable(exc) else "blocked", "collector_claim_unavailable"
        )
    if claim is None:
        return LocalOwnerDeleteAttempt("idle")
    progress = None

    def checkpoint(value):
        if not store.update_owner_delete_progress(claim, _canonical(value)):
            raise _Stale()

    try:
        root, jobs, db = map(_absolute, (artifact_root, jobs_root, jobs_db))
        scope = ManagedSessionScope(
            claim.task.environment_id, claim.task.device_id, claim.task.table_id
        )
        record = store.get_managed_resource_spec_for_session(scope, claim.session_id)
        if (
            Path(record.spec.paths["artifact_root"]) != root
            or Path(record.spec.paths["jobs_root"]) != jobs
            or db != jobs / "analysis-jobs.sqlite3"
            or store.artifact_path_for_session(claim.session_id) != root
        ):
            raise _Unsafe("original_root_mismatch")
        fs = _port(platform or os.name)
        with session_media_lock(root, claim.session_id, exclusive=True):
            # Reuse admission's immutable selection/fingerprint validation; exact
            # replay writes no authority and cannot adopt a changed saved clip.
            if store.accept_owner_delete_task(claim.task, managed_scope=scope) != claim:
                raise _Stale()
            if not store.update_owner_delete_progress(claim, claim.progress_json):
                raise _Stale()
            loaded = _json(claim.progress_json)
            specs, directories = _specs(
                fs,
                claim,
                db,
                root,
                jobs,
                progress=loaded,
                validate_payload=lambda payload: _owner_source_identity(
                    store, claim, scope, record, payload
                ),
            )
            roots = {"artifacts": root, "jobs": jobs}
            namespace = ".local-owner-media-cleanup"
            if loaded:
                _validate_progress(loaded, specs, roots, claim, namespace=namespace)
            progress = loaded
            _check_names(fs, directories, progress)
            if not progress:
                progress = _inventory(fs, specs, roots, claim, cancelled, namespace=namespace)
            checkpoint(progress)
            _preflight_disposal(fs, progress, roots, cancelled, checkpoint)
            _dispose(fs, progress, roots, cancelled, fault_hook, checkpoint)
            _preflight_disposal(fs, progress, roots, cancelled, checkpoint)
            if not store.complete_owner_delete(claim, progress_json=_canonical(progress)):
                raise _Stale()
        return LocalOwnerDeleteAttempt(
            "completed",
            reclaimed_bytes=sum(obj["reclaimed_bytes"] for obj in progress["objects"]),
            session_id=claim.session_id,
            shot_id=claim.shot_id,
        )
    except _Stale:
        return LocalOwnerDeleteAttempt(
            "stale", "claim_lost", session_id=claim.session_id, shot_id=claim.shot_id
        )
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        retry = _owner_retryable(exc)
        code = (
            str(exc)
            if isinstance(exc, _Unsafe)
            else ("media_busy" if retry else "unsafe_or_invalid_media")
        )
        try:
            encoded = None if progress is None else _canonical(progress)
        except _Unsafe:
            encoded = None
        try:
            if retry:
                accepted = store.retry_owner_delete(
                    claim,
                    code,
                    now + timedelta(seconds=min(2 ** min(claim.attempts, 8), 300)),
                    progress_json=encoded,
                )
            else:
                accepted = store.block_owner_delete(claim, code, progress_json=encoded)
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as terminal_exc:
            # A busy ledger cannot record retry yet. Keep its prior durable
            # journal/lease; the next claim recovers it after the lease expires.
            return LocalOwnerDeleteAttempt(
                "retry" if _owner_retryable(terminal_exc) else "blocked",
                "collector_checkpoint_unavailable",
                session_id=claim.session_id,
                shot_id=claim.shot_id,
            )
        return (
            LocalOwnerDeleteAttempt(
                "retry" if retry else "blocked",
                code,
                sum(obj.get("reclaimed_bytes", 0) for obj in (progress or {}).get("objects", [])),
                claim.session_id,
                claim.shot_id,
            )
            if accepted
            else LocalOwnerDeleteAttempt(
                "stale", "claim_lost", session_id=claim.session_id, shot_id=claim.shot_id
            )
        )
