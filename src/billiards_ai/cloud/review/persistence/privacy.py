"""Fail-closed privacy checks for authoritative persistence payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

_KEY_SEPARATOR = re.compile(r"[^a-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ALWAYS_SENSITIVE_KEY_PARTS = {
    "raw",
    "media",
    "video",
    "image",
    "locator",
    "file",
    "openid",
    "token",
    "secret",
    "password",
    "prompt",
    "provider",
    "url",
    "uri",
    "path",
}


def _key_parts(value: object) -> set[str]:
    separated = _CAMEL_BOUNDARY.sub("_", str(value).strip())
    return {part for part in _KEY_SEPARATOR.split(separated.lower()) if part}


def _is_sensitive_key(key: object, *, allow_reviewer_identity: bool) -> bool:
    parts = _key_parts(key)
    if not allow_reviewer_identity and "reviewer" in parts:
        return True
    if "openid" in parts or {"open", "id"} <= parts:
        return True
    return bool(parts & _ALWAYS_SENSITIVE_KEY_PARTS)


def assert_safe_persistence_payload(
    value: object,
    *,
    allow_reviewer_identity: bool = False,
    _path: str = "payload",
) -> None:
    """Reject sensitive material anywhere in a JSON-like payload.

    Public events must use the default. Private audit records may explicitly
    allow reviewer identity, while every other sensitive class remains denied.
    """

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _is_sensitive_key(key, allow_reviewer_identity=allow_reviewer_identity):
                raise ValueError(f"sensitive persistence field is forbidden at {_path}.{key}")
            assert_safe_persistence_payload(
                nested,
                allow_reviewer_identity=allow_reviewer_identity,
                _path=f"{_path}.{key}",
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            assert_safe_persistence_payload(
                nested,
                allow_reviewer_identity=allow_reviewer_identity,
                _path=f"{_path}[{index}]",
            )
        return
    if isinstance(value, str) and ("://" in value or value.startswith(("/", "~/"))):
        raise ValueError(f"sensitive locator value is forbidden at {_path}")
