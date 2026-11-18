"""Controlled local object-store adapter used by the Phase 0 deletion runner."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath

from .retention_repository import RetentionRepositoryError


class ControlledFilesystemMediaStore:
    """Delete only manifest locators contained by one explicit private root."""

    ENVIRONMENT_VARIABLE = "BILLIARDS_MEDIA_ROOT"
    SENTINEL_NAME = ".billiards-media-root"
    SENTINEL_CONTENT = "billiardsAI-media-root-v1\n"

    def __init__(self, root: Path | str):
        self.root = Path(root)
        if not self.root.exists():
            raise RetentionRepositoryError("media root must already exist")
        info = self.root.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
        ):
            raise RetentionRepositoryError("media root must be a real owned directory")
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise RetentionRepositoryError("media root permissions must be exactly 0700")
        self._root_identity = (info.st_dev, info.st_ino)
        self._resolved_root = self.root.resolve(strict=True)
        home = Path.home().resolve(strict=False)
        cwd = Path.cwd().resolve(strict=True)
        if (
            self._resolved_root == Path("/")
            or self._resolved_root.parent == Path("/")
            or self._resolved_root == home
            or self._resolved_root.parent == home
            or self._resolved_root == cwd
            or self._resolved_root.parent == cwd
        ):
            raise RetentionRepositoryError("media root is too broad")
        sentinel = self.root / self.SENTINEL_NAME
        try:
            sentinel_info = sentinel.lstat()
            if (
                stat.S_ISLNK(sentinel_info.st_mode)
                or not stat.S_ISREG(sentinel_info.st_mode)
                or sentinel_info.st_uid != os.geteuid()
                or stat.S_IMODE(sentinel_info.st_mode) != 0o600
                or sentinel.read_text(encoding="utf-8") != self.SENTINEL_CONTENT
            ):
                raise RetentionRepositoryError("media root sentinel is invalid")
        except FileNotFoundError as exc:
            raise RetentionRepositoryError("media root sentinel is required") from exc

    @classmethod
    def from_environment(cls) -> ControlledFilesystemMediaStore:
        root = os.environ.get(cls.ENVIRONMENT_VARIABLE)
        if not root:
            raise RetentionRepositoryError(
                f"{cls.ENVIRONMENT_VARIABLE} must identify the controlled media root"
            )
        return cls(root)

    @staticmethod
    def _locator_parts(locator: str) -> tuple[str, ...]:
        logical = PurePosixPath(locator)
        if not locator.strip() or logical.is_absolute() or ".." in logical.parts:
            raise RetentionRepositoryError("media locator must be a safe relative path")
        if not logical.parts:
            raise RetentionRepositoryError("media locator must identify an object")
        return logical.parts

    def _open_parent(self, locator: str) -> tuple[int, str]:
        parts = self._locator_parts(locator)
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        current_fd = -1
        try:
            current_fd = os.open(self.root, flags)
            root_info = os.fstat(current_fd)
            if (root_info.st_dev, root_info.st_ino) != self._root_identity:
                raise RetentionRepositoryError("media root changed after validation")
            for part in parts[:-1]:
                next_fd = os.open(part, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd, parts[-1]
        except RetentionRepositoryError:
            if current_fd >= 0:
                os.close(current_fd)
            raise
        except OSError as exc:
            if current_fd >= 0:
                os.close(current_fd)
            raise RetentionRepositoryError("media locator path is not trusted") from exc

    def delete(self, locator: str) -> None:
        parent_fd, leaf = self._open_parent(locator)
        try:
            try:
                info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise RetentionRepositoryError("media locator is not a regular object")
            os.unlink(leaf, dir_fd=parent_fd)
        except RetentionRepositoryError:
            raise
        except OSError as exc:
            raise RetentionRepositoryError("media object could not be deleted") from exc
        finally:
            os.close(parent_fd)

    def exists(self, locator: str) -> bool:
        try:
            parent_fd, leaf = self._open_parent(locator)
        except RetentionRepositoryError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return False
            raise
        try:
            info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        finally:
            os.close(parent_fd)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RetentionRepositoryError("media locator is not a regular object")
        return True
