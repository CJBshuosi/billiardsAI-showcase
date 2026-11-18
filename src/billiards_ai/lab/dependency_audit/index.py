"""PEP 691 wheel parsing and deterministic target-specific selection."""

import json
import re
from collections.abc import Iterable
from urllib.parse import unquote_to_bytes, urlsplit

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, compatible_tags, cpython_tags, mac_platforms
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import Version
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit.models import (
    CandidateDiagnostic,
    MarkerEnvironment,
    TargetPlatform,
    TargetSnapshot,
    WheelArtifact,
    WheelCandidateIndex,
)

_Target = TargetPlatform | TargetSnapshot


class WheelIndexError(ValueError):
    """Raised when metadata cannot safely yield a target-specific wheel."""

    def __init__(
        self,
        message: str,
        diagnostics: tuple[CandidateDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _target_parts(target: _Target) -> tuple[int, int, int]:
    if target.os not in {"macos", "windows", "linux"}:
        raise WheelIndexError(f"unsupported target operating system: {target.os}")
    if target.architecture not in {"arm64", "x86_64"}:
        raise WheelIndexError(f"unsupported target architecture: {target.architecture}")
    if target.python_implementation != "cp":
        raise WheelIndexError(
            f"unsupported target Python implementation: {target.python_implementation}"
        )
    try:
        python_parts = tuple(int(part) for part in target.python_version.split("."))
        os_parts = tuple(int(part) for part in target.os_version.split("."))
    except ValueError as error:
        raise WheelIndexError("target versions must contain only integers") from error
    if len(python_parts) != 3 or python_parts[0] != 3 or not os_parts:
        raise WheelIndexError("unsupported target version combination")
    return python_parts


def _platforms(target: _Target) -> tuple[str, ...]:
    _target_parts(target)
    if target.os == "macos":
        version = tuple(int(part) for part in target.os_version.split("."))
        return tuple(mac_platforms(version=version, arch=target.architecture))
    if target.os == "windows":
        architecture = "arm64" if target.architecture == "arm64" else "amd64"
        return (f"win_{architecture}",)
    architecture = "aarch64" if target.architecture == "arm64" else "x86_64"
    return (
        f"manylinux_2_17_{architecture}",
        f"manylinux2014_{architecture}",
        f"linux_{architecture}",
    )


def target_tags(target: _Target) -> tuple[Tag, ...]:
    """Return the ordered, immutable compatibility tags for the frozen target."""

    python_parts = _target_parts(target)
    python_version = python_parts[:2]
    platforms = _platforms(target)
    tags = tuple(cpython_tags(python_version=python_version, platforms=platforms)) + tuple(
        compatible_tags(
            python_version=python_version,
            interpreter=f"cp{python_version[0]}{python_version[1]}",
            platforms=platforms,
        )
    )
    return tuple(dict.fromkeys(tags))


def _marker_environment(target: TargetPlatform) -> MarkerEnvironment:
    python_parts = _target_parts(target)
    machine_by_target = {
        ("macos", "arm64"): "arm64",
        ("macos", "x86_64"): "x86_64",
        ("windows", "arm64"): "ARM64",
        ("windows", "x86_64"): "AMD64",
        ("linux", "arm64"): "aarch64",
        ("linux", "x86_64"): "x86_64",
    }
    os_values = {
        "macos": ("posix", "Darwin", "darwin"),
        "windows": ("nt", "Windows", "win32"),
        "linux": ("posix", "Linux", "linux"),
    }
    try:
        os_name, platform_system, sys_platform = os_values[target.os]
        platform_machine = machine_by_target[(target.os, target.architecture)]
    except KeyError as error:
        raise WheelIndexError("unsupported target marker combination") from error
    return MarkerEnvironment(
        implementation_name="cpython",
        implementation_version=target.python_version,
        os_name=os_name,
        platform_machine=platform_machine,
        platform_python_implementation="CPython",
        platform_release=None,
        platform_system=platform_system,
        platform_version=None,
        python_full_version=target.python_version,
        python_version=f"{python_parts[0]}.{python_parts[1]}",
        sys_platform=sys_platform,
    )


def resolve_target(target: TargetPlatform) -> TargetSnapshot:
    """Resolve one immutable audit context from a validated request target."""

    return TargetSnapshot(
        os=target.os,
        os_version=target.os_version,
        architecture=target.architecture,
        python_implementation=target.python_implementation,
        python_version=target.python_version,
        wheel_tags=tuple(str(tag) for tag in target_tags(target)),
        marker_environment=_marker_environment(target),
    )


def _snapshot_tags(target: TargetSnapshot) -> tuple[Tag, ...]:
    if len(target.wheel_tags) != len(set(target.wheel_tags)):
        raise WheelIndexError("target wheel tags must be unique")
    tags: list[Tag] = []
    for value in target.wheel_tags:
        parts = value.split("-", maxsplit=2)
        if len(parts) != 3 or not all(parts):
            raise WheelIndexError(f"target wheel tag is invalid: {value}")
        tags.append(Tag(*parts))
    return tuple(tags)


def _diagnostic(
    package: str,
    filename: str,
    category: str,
    reason: str,
    detail: str | None = None,
) -> CandidateDiagnostic:
    return CandidateDiagnostic(
        package=canonicalize_name(package),
        filename=filename,
        category=category,
        reason=reason,
        detail=detail,
    )


def _blocked(
    package: str,
    filename: str,
    reason: str,
    detail: str,
) -> WheelIndexError:
    diagnostic = _diagnostic(package, filename, "blocked", reason, detail)
    return WheelIndexError(
        f"package {package} file {filename}: {detail}",
        (diagnostic,),
    )


def _requires_python(
    record: dict[str, object],
    target: TargetSnapshot,
    package: str,
    filename: str,
) -> tuple[str | None, CandidateDiagnostic | None]:
    value = record.get("requires-python")
    if value is None:
        return None, None
    if not isinstance(value, str):
        raise _blocked(
            package,
            filename,
            "malformed_requires_python",
            "requires-python must be a string",
        )
    try:
        requirement = SpecifierSet(value)
    except InvalidSpecifier as error:
        raise _blocked(
            package,
            filename,
            "malformed_requires_python",
            "requires-python is invalid",
        ) from error
    if Version(target.python_version) not in requirement:
        return value, _diagnostic(
            package,
            filename,
            "excluded",
            "requires_python",
            f"target Python {target.python_version} does not satisfy {value}",
        )
    return value, None


def _validate_archive_url(url: str, filename: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("archive URL must use https")
    if parsed.netloc != "files.pythonhosted.org" or parsed.hostname != "files.pythonhosted.org":
        raise ValueError("archive host is not allowed")
    if parsed.username is not None or parsed.password is not None or parsed.port is not None:
        raise ValueError("archive URL credentials and ports are not allowed")
    if parsed.query or parsed.fragment:
        raise ValueError("archive URL query strings and fragments are not allowed")
    if re.search(r"%(?![0-9A-Fa-f]{2})", parsed.path):
        raise ValueError("archive URL contains invalid percent encoding")
    lowered_path = parsed.path.lower()
    if "%2f" in lowered_path or "%5c" in lowered_path:
        raise ValueError("archive URL encoded separators are not allowed")
    decoded_path = unquote_to_bytes(parsed.path).decode("utf-8", errors="strict")
    if "\\" in decoded_path or any(
        ord(character) < 32 or ord(character) == 127 for character in decoded_path
    ):
        raise ValueError("archive path contains unsafe characters")
    segments = decoded_path.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("archive path contains dot segments")
    if (
        re.fullmatch(
            r"/packages/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{60}/[^/]+",
            decoded_path,
        )
        is None
    ):
        raise ValueError("archive path is not allowed")
    if segments[-1] != filename:
        raise ValueError("archive URL basename does not match wheel filename")


def _selectable_artifact(
    record: object,
    package: str,
    target: TargetSnapshot,
    frozen_tags: tuple[Tag, ...],
) -> tuple[WheelArtifact | None, CandidateDiagnostic | None]:
    if not isinstance(record, dict):
        raise _blocked(
            package,
            "<missing filename>",
            "malformed_record",
            "file record must be an object",
        )
    raw_filename = record.get("filename")
    if not isinstance(raw_filename, str) or not raw_filename:
        raise _blocked(
            package,
            "<missing filename>",
            "malformed_record",
            "filename is required",
        )
    if not raw_filename.endswith(".whl"):
        return None, _diagnostic(
            package,
            raw_filename,
            "excluded",
            "source_distribution",
            "source and non-wheel distributions are not selectable",
        )
    try:
        wheel_name, version, _, wheel_tags = parse_wheel_filename(raw_filename)
    except InvalidWheelFilename as error:
        raise _blocked(
            package,
            raw_filename,
            "malformed_filename",
            "wheel filename is invalid",
        ) from error
    if canonicalize_name(str(wheel_name)) != canonicalize_name(package):
        raise _blocked(
            package,
            raw_filename,
            "package_name_mismatch",
            "wheel package name does not match metadata",
        )
    if version.is_prerelease:
        return None, _diagnostic(
            package,
            raw_filename,
            "excluded",
            "prerelease",
            f"version {version} is a prerelease",
        )
    matching_tags = tuple(str(tag) for tag in frozen_tags if tag in wheel_tags)
    if not matching_tags:
        return None, _diagnostic(
            package,
            raw_filename,
            "excluded",
            "incompatible_tag",
            "wheel tags do not intersect the frozen target tag order",
        )
    yanked = record.get("yanked")
    if yanked is False:
        pass
    elif yanked is True:
        return None, _diagnostic(
            package,
            raw_filename,
            "excluded",
            "yanked",
        )
    elif isinstance(yanked, str) and yanked:
        return None, _diagnostic(
            package,
            raw_filename,
            "excluded",
            "yanked",
            yanked,
        )
    else:
        raise _blocked(
            package,
            raw_filename,
            "malformed_yanked",
            "yanked must be a boolean or a non-empty reason string",
        )
    requires_python, python_diagnostic = _requires_python(record, target, package, raw_filename)
    if python_diagnostic is not None:
        return None, python_diagnostic
    hashes = record.get("hashes")
    sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise _blocked(
            package,
            raw_filename,
            "missing_sha256",
            "sha256 is required",
        )
    size = record.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise _blocked(
            package,
            raw_filename,
            "invalid_size",
            "size must be a positive integer",
        )
    url = record.get("url")
    if not isinstance(url, str) or not url:
        raise _blocked(
            package,
            raw_filename,
            "invalid_archive_url",
            "archive URL is required",
        )
    try:
        _validate_archive_url(url, raw_filename)
    except (ValueError, UnicodeError) as error:
        raise _blocked(
            package,
            raw_filename,
            "invalid_archive_url",
            str(error),
        ) from error
    upload_time = record.get("upload-time")
    if not isinstance(upload_time, str) or not upload_time:
        raise _blocked(
            package,
            raw_filename,
            "malformed_metadata",
            "upload-time is required",
        )
    try:
        artifact = WheelArtifact(
            name=canonicalize_name(package),
            version=str(version),
            filename=raw_filename,
            url=url,
            tags=matching_tags,
            bytes=size,
            sha256=sha256,
            requires_python=requires_python,
            upload_time=upload_time,
            is_pure_python=all(tag.platform == "any" for tag in wheel_tags),
        )
    except ValidationError as error:
        raise _blocked(
            package,
            raw_filename,
            "malformed_metadata",
            "wheel metadata is invalid",
        ) from error
    return artifact, None


def _diagnostic_key(item: CandidateDiagnostic) -> tuple[str, str, str, str, str]:
    return (
        canonicalize_name(item.package),
        item.filename,
        item.category,
        item.reason,
        item.detail or "",
    )


def parse_simple_index(
    payload: bytes,
    target: TargetSnapshot,
    *,
    expected_package: str | None = None,
) -> WheelCandidateIndex:
    """Parse every PEP 691 record into a candidate or structured diagnostic."""

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WheelIndexError("invalid simple API JSON") from error
    if not isinstance(document, dict):
        raise WheelIndexError("simple API document must be an object")
    name = document.get("name")
    files = document.get("files")
    if not isinstance(name, str) or not name:
        raise WheelIndexError("simple API document has no package name")
    package = canonicalize_name(name)
    if expected_package is not None and package != canonicalize_name(expected_package):
        raise WheelIndexError(
            f"package {expected_package}: simple metadata package name does not match request"
        )
    if not isinstance(files, list):
        raise WheelIndexError(f"package {name}: files must be a list")

    frozen_tags = _snapshot_tags(target)
    tag_rank = {str(tag): position for position, tag in enumerate(frozen_tags)}
    artifacts: list[WheelArtifact] = []
    diagnostics: list[CandidateDiagnostic] = []
    for record in files:
        try:
            artifact, diagnostic = _selectable_artifact(record, package, target, frozen_tags)
        except WheelIndexError as error:
            diagnostics.extend(error.diagnostics)
            continue
        if artifact is not None:
            artifacts.append(artifact)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    sorted_diagnostics = tuple(
        {_diagnostic_key(item): item for item in diagnostics}[key]
        for key in sorted({_diagnostic_key(item) for item in diagnostics})
    )
    blocked = tuple(item for item in sorted_diagnostics if item.category == "blocked")
    if blocked:
        details = "; ".join(f"{item.filename}: {item.detail or item.reason}" for item in blocked)
        raise WheelIndexError(
            f"package {package}: blocked archive records: {details}",
            sorted_diagnostics,
        )
    artifacts.sort(key=lambda artifact: artifact.filename)
    artifacts.sort(key=lambda artifact: tag_rank[artifact.tags[0]])
    artifacts.sort(key=lambda artifact: Version(artifact.version), reverse=True)
    return WheelCandidateIndex(
        candidates=tuple(artifacts),
        diagnostics=sorted_diagnostics,
    )


def parse_simple_candidates(payload: bytes, target: _Target) -> tuple[WheelArtifact, ...]:
    """Compatibility wrapper returning stable compatible wheels only."""

    snapshot = target if isinstance(target, TargetSnapshot) else resolve_target(target)
    return parse_simple_index(payload, snapshot).candidates


def select_best_wheel(
    candidates: Iterable[WheelArtifact], requirements: SpecifierSet
) -> WheelArtifact:
    """Select the first matching candidate from parser-established deterministic order."""

    for candidate in candidates:
        version = Version(candidate.version)
        if not version.is_prerelease and version in requirements:
            return candidate
    raise WheelIndexError("no compatible stable wheel matches requirements")
