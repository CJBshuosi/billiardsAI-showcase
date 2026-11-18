from hashlib import md5

import pytest

from billiards_ai.lab.dependency_audit.models import TargetPlatform
from billiards_ai.lab.pooltool_audit.archive import (
    ArchiveIndexError,
    PandaArchiveHead,
    select_panda3d_wheel,
)

TARGET_URL = (
    "https://buildbot.panda3d.org/downloads/"
    "8114971c56ce89781e58336be6d5b349aca07b5a/"
    "panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl"
)
TARGET_MD5 = "d6763ec4bdae49bf34527a30d8a69424"


def _target() -> TargetPlatform:
    return TargetPlatform(
        os="windows",
        os_version="11.0",
        architecture="x86_64",
        python_implementation="cp",
        python_version="3.12.10",
    )


def _html(*links: str) -> bytes:
    anchors = "".join(f'<a href="{link}">wheel</a>' for link in links)
    return f"<html><body>{anchors}</body></html>".encode()


def test_selects_only_exact_pinned_windows_cp312_wheel() -> None:
    other = TARGET_URL.replace("cp312-cp312-win_amd64", "cp311-cp311-win_amd64")
    payload = _html(f"{other}#md5={'0' * 32}", f"{TARGET_URL}#md5={TARGET_MD5}")
    artifact = select_panda3d_wheel(
        payload,
        target=_target(),
        version="1.11.0.dev3702",
        head=PandaArchiveHead(
            url=TARGET_URL,
            bytes=53_836_581,
            last_modified="Wed, 29 Jan 2025 21:05:00 GMT",
            etag='"3357b25-62cdea999d9c4"',
        ),
    )
    assert artifact.filename.endswith("cp312-cp312-win_amd64.whl")
    assert artifact.md5 == TARGET_MD5
    assert artifact.bytes == 53_836_581
    assert artifact.sha256 is None
    assert artifact.approval_status == "sha256_required"


def test_rejects_unapproved_archive_host() -> None:
    payload = _html(
        "https://evil.example/downloads/8114971/"
        f"panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl#md5={TARGET_MD5}"
    )
    with pytest.raises(ArchiveIndexError, match="archive host"):
        select_panda3d_wheel(
            payload,
            target=_target(),
            version="1.11.0.dev3702",
            head=PandaArchiveHead(
                url=TARGET_URL,
                bytes=1,
                last_modified="now",
                etag='"etag"',
            ),
        )


def test_rejects_head_url_mismatch() -> None:
    payload = _html(f"{TARGET_URL}#md5={TARGET_MD5}")
    with pytest.raises(ArchiveIndexError, match="HEAD URL"):
        select_panda3d_wheel(
            payload,
            target=_target(),
            version="1.11.0.dev3702",
            head=PandaArchiveHead(
                url=TARGET_URL + ".other",
                bytes=1,
                last_modified="now",
                etag='"etag"',
            ),
        )


def test_published_md5_is_not_mislabeled_as_sha256() -> None:
    payload = _html(f"{TARGET_URL}#md5={md5(b'not-the-wheel').hexdigest()}")
    artifact = select_panda3d_wheel(
        payload,
        target=_target(),
        version="1.11.0.dev3702",
        head=PandaArchiveHead(
            url=TARGET_URL,
            bytes=53_836_581,
            last_modified="Wed, 29 Jan 2025 21:05:00 GMT",
            etag='"etag"',
        ),
    )
    assert artifact.sha256 is None
