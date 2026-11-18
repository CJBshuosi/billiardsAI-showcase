"""Bounded, crash-safe frame evidence spool for Mainline 1 live capture."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import numpy as np

from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame


class LiveBackpressureError(RuntimeError):
    """The live epoch reached its explicit frame capacity."""


class LiveSpoolContractError(ValueError):
    """Frame or segment evidence violates the ordered spool contract."""


class LiveCaptureState(StrEnum):
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    CLOSED = "CLOSED"


@dataclass(frozen=True, slots=True)
class LiveCaptureHealth:
    state: LiveCaptureState
    stored_frames: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SpoolSegment:
    shot_id: str
    path: Path
    start_frame_id: int
    end_frame_id: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    frame_count: int
    byte_count: int
    sha256: str
    complete: bool


class CaptureSpool:
    """Persist an ordered live epoch without silently dropping frames."""

    def __init__(self, root: Path, *, maximum_frames: int) -> None:
        if maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise LiveSpoolContractError("spool root must be a directory")
        self._owner_path = self._root / ".capture-spool-owner.json"
        self._owner_token = uuid4().hex
        self._owns_root = False
        self._acquire_owner()
        try:
            for pending in self._root.glob(".pending-*"):
                if pending.is_dir() and not pending.is_symlink():
                    shutil.rmtree(pending)
            self._frames_root = self._root / ".live-frames"
            if self._frames_root.is_symlink():
                raise LiveSpoolContractError("live frame spool must not be a symlink")
            if self._frames_root.exists():
                recovered = self._root / f".recovered-live-frames-{uuid4().hex}"
                os.replace(self._frames_root, recovered)
            self._frames_root.mkdir()
            self._maximum_frames = maximum_frames
            self._frames: dict[int, DecodedTrackingFrame] = {}
            self._state = LiveCaptureState.RUNNING
            self._reason: str | None = None
            self._last_frame_id: int | None = None
            self._last_timestamp_ms: int | None = None
        except BaseException:
            self._release_owner()
            raise

    @property
    def health(self) -> LiveCaptureHealth:
        return LiveCaptureHealth(
            state=self._state,
            stored_frames=len(self._frames),
            reason=self._reason,
        )

    def append(self, frame: DecodedTrackingFrame) -> None:
        if self._state is LiveCaptureState.CLOSED:
            raise LiveSpoolContractError("spool is closed")
        if self._state is LiveCaptureState.DEGRADED:
            raise LiveBackpressureError(self._reason or "live spool is degraded")
        if self._last_frame_id is not None and frame.frame_id <= self._last_frame_id:
            raise LiveSpoolContractError("frame_id must be strictly increasing")
        if self._last_timestamp_ms is not None and frame.timestamp_ms < self._last_timestamp_ms:
            raise LiveSpoolContractError("frame timestamp must be monotonic")
        if frame.image.ndim != 3 or frame.image.shape[2] != 3:
            raise LiveSpoolContractError("frame image must be BGR HxWx3")
        if len(self._frames) >= self._maximum_frames:
            self._state = LiveCaptureState.DEGRADED
            self._reason = "live spool capacity exhausted"
            raise LiveBackpressureError(self._reason)

        destination = self._frame_path(frame.frame_id)
        temporary = self._frames_root / f".pending-frame-{uuid4().hex}.npy"
        try:
            with temporary.open("wb") as stream:
                np.save(stream, frame.image, allow_pickle=False)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        stored = DecodedTrackingFrame(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            image=frame.image.copy(),
        )
        self._frames[frame.frame_id] = stored
        self._last_frame_id = frame.frame_id
        self._last_timestamp_ms = frame.timestamp_ms

    def frames_between(
        self, start_frame_id: int, end_frame_id: int
    ) -> tuple[DecodedTrackingFrame, ...]:
        if end_frame_id < start_frame_id:
            raise LiveSpoolContractError("frame range is reversed")
        selected = tuple(
            frame
            for frame_id, frame in sorted(self._frames.items())
            if start_frame_id <= frame_id <= end_frame_id
        )
        return tuple(
            DecodedTrackingFrame(
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                image=frame.image.copy(),
            )
            for frame in selected
        )

    def seal_shot(
        self,
        *,
        shot_id: str,
        start_frame_id: int,
        end_frame_id: int,
        complete: bool,
    ) -> SpoolSegment:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", shot_id):
            raise LiveSpoolContractError("shot_id is invalid")
        frames = self.frames_between(start_frame_id, end_frame_id)
        if not frames:
            raise LiveSpoolContractError("shot range contains no frames")
        expected_ids = tuple(range(start_frame_id, end_frame_id + 1))
        actual_ids = tuple(frame.frame_id for frame in frames)
        if complete and actual_ids != expected_ids:
            raise LiveSpoolContractError("complete shot frame range must be contiguous")
        destination = self._root / shot_id
        if destination.exists():
            raise LiveSpoolContractError("shot segment already exists")
        temporary = self._root / f".pending-{shot_id}-{uuid4().hex}"
        try:
            temporary.mkdir()
            archive = temporary / "frames.npz"
            with archive.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    frame_ids=np.asarray(actual_ids, dtype=np.int64),
                    timestamps_ms=np.asarray(
                        [frame.timestamp_ms for frame in frames], dtype=np.int64
                    ),
                    images=np.stack([frame.image for frame in frames]),
                )
            digest = self._sha256(archive)
            byte_count = archive.stat().st_size
            manifest = {
                "byte_count": byte_count,
                "complete": complete,
                "end_frame_id": frames[-1].frame_id,
                "end_timestamp_ms": frames[-1].timestamp_ms,
                "frame_count": len(frames),
                "frames_sha256": digest,
                "schema_version": "1.0",
                "shot_id": shot_id,
                "start_frame_id": frames[0].frame_id,
                "start_timestamp_ms": frames[0].timestamp_ms,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        # The settled end frame is also the stable baseline immediately before a
        # possible next shot. Preserve it until the following segment is sealed.
        for frame_id in tuple(self._frames):
            if frame_id < end_frame_id:
                self._frames.pop(frame_id)
                frame_path = self._frame_path(frame_id)
                if frame_path.exists():
                    frame_path.unlink()
        return SpoolSegment(
            shot_id=shot_id,
            path=destination,
            start_frame_id=frames[0].frame_id,
            end_frame_id=frames[-1].frame_id,
            start_timestamp_ms=frames[0].timestamp_ms,
            end_timestamp_ms=frames[-1].timestamp_ms,
            frame_count=len(frames),
            byte_count=byte_count,
            sha256=digest,
            complete=complete,
        )

    def close(self) -> None:
        self._state = LiveCaptureState.CLOSED
        self._reason = None
        self._release_owner()

    def _acquire_owner(self) -> None:
        payload = json.dumps(
            {"pid": os.getpid(), "token": self._owner_token},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        for _ in range(3):
            try:
                descriptor = os.open(
                    self._owner_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                if not self._recover_stale_owner():
                    raise LiveSpoolContractError("spool root is already owned") from None
                continue
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._owns_root = True
            return
        raise LiveSpoolContractError("could not acquire spool root ownership")

    def _recover_stale_owner(self) -> bool:
        if self._owner_path.is_symlink():
            return False
        try:
            owner = json.loads(self._owner_path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return False
        if _process_is_alive(pid):
            return False
        try:
            self._owner_path.unlink()
        except FileNotFoundError:
            pass
        return True

    def _release_owner(self) -> None:
        if not self._owns_root:
            return
        try:
            owner = json.loads(self._owner_path.read_text(encoding="utf-8"))
            if owner.get("token") == self._owner_token:
                self._owner_path.unlink(missing_ok=True)
        except (OSError, AttributeError, json.JSONDecodeError):
            pass
        finally:
            self._owns_root = False

    def _frame_path(self, frame_id: int) -> Path:
        return self._frames_root / f"{frame_id:020d}.npy"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
