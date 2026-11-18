"""Shared group-first ball identity label contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

BallGroup = Literal["cue", "solid", "eight", "stripe", "unknown"]
ExactBallIdentity = Annotated[str, Field(pattern=r"^(cue|[1-9]|1[0-5])$")]
ExactIdentityEvidence = Literal[
    "group_intrinsic",
    "number_visible",
    "track_confirmed",
    "known_set_mapping",
]

_GROUPS = {"cue", "solid", "eight", "stripe", "unknown"}
_EVIDENCE = {
    "group_intrinsic",
    "number_visible",
    "track_confirmed",
    "known_set_mapping",
}


def validate_identity_label(
    *,
    group: str,
    exact_identity: str | None,
    exact_evidence: str | None,
) -> None:
    """Reject inconsistent group/exact/evidence combinations."""

    if group not in _GROUPS:
        raise ValueError("identity group is invalid")
    if exact_evidence is not None and exact_evidence not in _EVIDENCE:
        raise ValueError("exact identity evidence is invalid")
    if exact_identity is not None and exact_identity not in {
        "cue",
        *(str(number) for number in range(1, 16)),
    }:
        raise ValueError("exact identity is invalid")

    if group == "cue":
        if exact_identity != "cue" or exact_evidence != "group_intrinsic":
            raise ValueError("cue group requires intrinsic cue identity evidence")
        return
    if group == "eight":
        if exact_identity != "8" or exact_evidence != "group_intrinsic":
            raise ValueError("eight group requires intrinsic eight identity evidence")
        return
    if group == "unknown":
        if exact_identity is not None or exact_evidence is not None:
            raise ValueError("unknown group cannot carry exact identity evidence")
        return
    if exact_identity is None:
        if exact_evidence is not None:
            raise ValueError("exact identity evidence requires an exact identity")
        return
    if exact_evidence is None:
        raise ValueError("exact identity requires evidence")
    if exact_evidence == "group_intrinsic":
        raise ValueError("object-ball exact identity cannot use group-intrinsic evidence")
    number = int(exact_identity) if exact_identity != "cue" else 0
    expected_group = "solid" if 1 <= number <= 7 else "stripe" if 9 <= number <= 15 else None
    if expected_group != group:
        raise ValueError("exact identity conflicts with identity group")


def migrate_v1_exact_identity(value: str) -> tuple[BallGroup, str | None, str | None]:
    """Map the legacy forced-exact label into the hierarchical contract."""

    if value == "cue":
        return "cue", "cue", "group_intrinsic"
    if value == "8":
        return "eight", "8", "group_intrinsic"
    if value == "unknown":
        return "unknown", None, None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("legacy exact identity is invalid") from exc
    if not 1 <= number <= 15 or number == 8:
        raise ValueError("legacy exact identity is invalid")
    group: BallGroup = "solid" if number <= 7 else "stripe"
    return group, str(number), "known_set_mapping"
