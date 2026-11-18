from __future__ import annotations

import pytest

from billiards_ai.lab.training.identity_labels import (
    migrate_v1_exact_identity,
    validate_identity_label,
)


def test_group_only_solid_label_is_valid() -> None:
    validate_identity_label(group="solid", exact_identity=None, exact_evidence=None)


def test_exact_number_requires_consistent_group_and_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        validate_identity_label(group="solid", exact_identity="4", exact_evidence=None)
    with pytest.raises(ValueError, match="group"):
        validate_identity_label(
            group="stripe",
            exact_identity="4",
            exact_evidence="number_visible",
        )


def test_cue_eight_and_unknown_are_intrinsic_or_unresolved() -> None:
    validate_identity_label(group="cue", exact_identity="cue", exact_evidence="group_intrinsic")
    validate_identity_label(group="eight", exact_identity="8", exact_evidence="group_intrinsic")
    validate_identity_label(group="unknown", exact_identity=None, exact_evidence=None)


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("cue", ("cue", "cue", "group_intrinsic")),
        ("8", ("eight", "8", "group_intrinsic")),
        ("4", ("solid", "4", "known_set_mapping")),
        ("12", ("stripe", "12", "known_set_mapping")),
        ("unknown", ("unknown", None, None)),
    ],
)
def test_v1_exact_identity_migration(
    legacy: str, expected: tuple[str, str | None, str | None]
) -> None:
    assert migrate_v1_exact_identity(legacy) == expected
