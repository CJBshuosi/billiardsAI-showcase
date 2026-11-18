from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

from billiards_ai.lab.dependency_audit.index import (
    WheelIndexError,
    parse_simple_candidates,
    parse_simple_index,
    resolve_target,
    select_best_wheel,
    target_tags,
)
from billiards_ai.lab.dependency_audit.models import TargetPlatform


@pytest.fixture
def target() -> TargetPlatform:
    return TargetPlatform(
        os="macos",
        os_version="14.0",
        architecture="arm64",
        python_implementation="cp",
        python_version="3.12.13",
    )


def _fixture(name: str) -> bytes:
    return (Path(__file__).parents[3] / "fixtures" / "pypi" / name).read_bytes()


def test_target_tags_include_the_exact_cpython_macos_arm64_tag(target: TargetPlatform) -> None:
    tags = target_tags(target)

    assert str(tags[0]) == "cp312-cp312-macosx_14_0_arm64"
    assert "cp311-cp311-macosx_14_0_arm64" not in {str(tag) for tag in tags}


def test_index_selects_macos_arm64_cp312_and_never_sdist(target: TargetPlatform) -> None:
    candidates = parse_simple_candidates(_fixture("demo-simple.json"), target)
    selected = select_best_wheel(candidates, SpecifierSet(">=1,<2"))

    assert selected.filename == "demo-1.2.0-cp312-cp312-macosx_14_0_arm64.whl"
    assert selected.bytes == 123456
    assert selected.sha256 == "1" * 64
    assert all(candidate.filename.endswith(".whl") for candidate in candidates)
    assert all(
        candidate.filename != "demo-1.3.0-cp312-cp312-macosx_14_0_arm64.whl"
        for candidate in candidates
    )


def test_index_rejects_missing_sha256_with_package_and_filename(target: TargetPlatform) -> None:
    payload = _fixture("demo-simple.json").replace(b'"sha256":"111', b'"md5":"111')

    with pytest.raises(
        WheelIndexError,
        match=r"demo.*demo-1\.2\.0-cp312-cp312-macosx_14_0_arm64\.whl.*sha256",
    ):
        parse_simple_candidates(payload, target)


def test_index_rejects_non_positive_size_on_a_selectable_wheel(target: TargetPlatform) -> None:
    payload = _fixture("demo-simple.json").replace(b'"size":123456', b'"size":0')

    with pytest.raises(
        WheelIndexError,
        match=r"demo.*demo-1\.2\.0-cp312-cp312-macosx_14_0_arm64\.whl.*size",
    ):
        parse_simple_candidates(payload, target)


def test_index_excludes_boolean_true_yanked_wheel_without_reason(
    target: TargetPlatform,
) -> None:
    filename = "demo-1.2.0-cp312-cp312-macosx_14_0_arm64.whl"
    payload = _fixture("demo-simple.json").replace(b'"yanked":false', b'"yanked":true', 1)

    result = parse_simple_index(payload, resolve_target(target))

    assert all(candidate.filename != filename for candidate in result.candidates)
    assert next(item for item in result.diagnostics if item.filename == filename).model_dump() == {
        "package": "demo",
        "filename": filename,
        "category": "excluded",
        "reason": "yanked",
        "detail": None,
    }


@pytest.mark.parametrize(
    "encoded_value",
    [b"null", b'""', b"[]", b"{}", b"0"],
    ids=["null", "empty-string", "list", "object", "number"],
)
def test_index_rejects_non_pep_691_yanked_value(
    target: TargetPlatform, encoded_value: bytes
) -> None:
    payload = _fixture("demo-simple.json").replace(
        b'"yanked":false', b'"yanked":' + encoded_value, 1
    )

    with pytest.raises(WheelIndexError) as captured:
        parse_simple_candidates(payload, target)

    assert any(
        item.filename == "demo-1.2.0-cp312-cp312-macosx_14_0_arm64.whl"
        and item.reason == "malformed_yanked"
        for item in captured.value.diagnostics
    )


def test_index_rejects_a_prerelease_even_when_it_matches_the_target(target: TargetPlatform) -> None:
    payload = _fixture("demo-simple.json").replace(
        b"demo-1.2.0-cp312-cp312-macosx_14_0_arm64.whl",
        b"demo-1.2.0rc1-cp312-cp312-macosx_14_0_arm64.whl",
        1,
    )
    candidates = parse_simple_candidates(payload, target)

    assert candidates == ()


def test_index_rejects_selection_when_no_stable_candidate_matches_requirement(
    target: TargetPlatform,
) -> None:
    candidates = parse_simple_candidates(_fixture("demo-simple.json"), target)

    with pytest.raises(WheelIndexError, match="no compatible stable wheel"):
        select_best_wheel(candidates, SpecifierSet(">=2"))
