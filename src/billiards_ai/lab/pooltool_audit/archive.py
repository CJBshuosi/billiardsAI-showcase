"""Narrow parser for the official Panda3D PEP 503 archive index."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit, urlunsplit

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import Version

from billiards_ai.lab.dependency_audit.index import resolve_target, target_tags
from billiards_ai.lab.dependency_audit.models import TargetPlatform
from billiards_ai.lab.pooltool_audit.models import PandaArchiveArtifact


class ArchiveIndexError(ValueError):
    """Raised when official archive metadata is incomplete or outside policy."""


@dataclass(frozen=True, slots=True)
class PandaArchiveHead:
    url: str
    bytes: int
    last_modified: str
    etag: str


@dataclass(frozen=True, slots=True)
class PandaArchiveLink:
    url: str
    filename: str
    md5: str
    tags: tuple[str, ...]


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = next((value for name, value in attrs if name.lower() == "href"), None)
        if href is not None:
            self.links.append(href)


def _validate_archive_url(url: str, filename: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "buildbot.panda3d.org":
        raise ArchiveIndexError("Panda3D archive host is not allowed")
    if parsed.netloc != "buildbot.panda3d.org" or parsed.username or parsed.password:
        raise ArchiveIndexError("Panda3D archive authority is not allowed")
    if parsed.port is not None or parsed.query:
        raise ArchiveIndexError("Panda3D archive URL query or port is not allowed")
    expected = rf"/downloads/[0-9a-f]{{40}}/{re.escape(filename)}"
    if re.fullmatch(expected, parsed.path) is None:
        raise ArchiveIndexError("Panda3D archive path is not allowed")


def _fragment_md5(url: str) -> str:
    parsed = urlsplit(url)
    values = parse_qs(parsed.fragment, strict_parsing=True)
    if set(values) != {"md5"} or len(values["md5"]) != 1:
        raise ArchiveIndexError("Panda3D archive link must publish exactly one MD5")
    value = values["md5"][0]
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ArchiveIndexError("Panda3D archive MD5 is invalid")
    return value


def _without_fragment(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def discover_panda3d_wheel(
    payload: bytes,
    *,
    target: TargetPlatform,
    version: str,
) -> PandaArchiveLink:
    """Discover the sole compatible exact-pinned wheel without reading its payload."""

    try:
        document = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchiveIndexError("Panda3D archive index is not UTF-8") from error
    parser = _LinkParser()
    parser.feed(document)
    frozen_tags = tuple(target_tags(resolve_target(target)))
    matches: list[tuple[str, str, tuple[str, ...]]] = []
    for link in parser.links:
        filename = urlsplit(link).path.rsplit("/", maxsplit=1)[-1]
        if not filename.startswith(f"panda3d-{version}-") or not filename.endswith(".whl"):
            continue
        _validate_archive_url(link, filename)
        published_md5 = _fragment_md5(link)
        try:
            name, parsed_version, _, wheel_tags = parse_wheel_filename(filename)
        except InvalidWheelFilename as error:
            raise ArchiveIndexError("Panda3D archive wheel filename is invalid") from error
        if str(name) != "panda3d" or parsed_version != Version(version):
            raise ArchiveIndexError("Panda3D archive wheel identity does not match")
        compatible = tuple(str(tag) for tag in frozen_tags if tag in wheel_tags)
        if compatible:
            matches.append((link, published_md5, compatible))
    if len(matches) != 1:
        raise ArchiveIndexError("Panda3D archive must yield exactly one compatible pinned wheel")
    link, published_md5, compatible = matches[0]
    wheel_url = _without_fragment(link)
    return PandaArchiveLink(
        url=wheel_url,
        filename=wheel_url.rsplit("/", maxsplit=1)[-1],
        md5=published_md5,
        tags=compatible,
    )


def select_panda3d_wheel(
    payload: bytes,
    *,
    target: TargetPlatform,
    version: str,
    head: PandaArchiveHead,
) -> PandaArchiveArtifact:
    """Select the wheel while retaining its explicit missing SHA-256 gate."""

    link = discover_panda3d_wheel(payload, target=target, version=version)
    wheel_url = link.url
    if head.url != wheel_url:
        raise ArchiveIndexError("Panda3D archive HEAD URL does not match selected wheel")
    if head.bytes <= 0 or not head.last_modified or not head.etag:
        raise ArchiveIndexError("Panda3D archive HEAD metadata is incomplete")
    return PandaArchiveArtifact(
        filename=link.filename,
        url=wheel_url,
        tags=link.tags,
        bytes=head.bytes,
        md5=link.md5,
        sha256=None,
        last_modified=head.last_modified,
        etag=head.etag,
        approval_status="sha256_required",
    )
