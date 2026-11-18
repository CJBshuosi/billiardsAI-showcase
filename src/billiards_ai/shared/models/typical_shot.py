"""Strict public contract for deterministic typical-shot selection."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import UUID4, BaseModel, ConfigDict, Field, StrictInt, model_validator


class TypicalShotStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy through validation so public contracts cannot be silently weakened."""

        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_model_values(copied))


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


class TypicalShotRuleVersion(StrEnum):
    V1 = "1.0"


class TypicalShotSelectionReason(StrEnum):
    SEVERE_FOUL = "severe_foul"
    SEVERE_DEVIATION = "severe_deviation"
    SEVERE_MISSED_POT = "severe_missed_pot"
    REPEATED_FOUL = "repeated_foul"
    REPEATED_LARGE_DEVIATION = "repeated_large_deviation"
    REPEATED_MISSED_POT = "repeated_missed_pot"
    POSITIVE_POTTING = "positive_potting"

    @property
    def category(self) -> str:
        if self.value.startswith("severe_"):
            return "severe"
        if self.value.startswith("repeated_"):
            return "repeated"
        return "positive"


class TypicalShotSelection(TypicalShotStrictFrozenModel):
    shot_id: UUID4
    sequence: StrictInt = Field(ge=1)
    selection_rank: StrictInt = Field(ge=1)
    selection_reason: TypicalShotSelectionReason
    rule_version: TypicalShotRuleVersion


class TypicalShotSelectionResult(TypicalShotStrictFrozenModel):
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    rule_version: TypicalShotRuleVersion
    selections: tuple[TypicalShotSelection, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def selections_are_safe_ranked_and_diverse(self) -> TypicalShotSelectionResult:
        selections = tuple(
            TypicalShotSelection.model_validate(_stored_model_values(selection))
            for selection in self.selections
        )
        if [selection.selection_rank for selection in selections] != list(
            range(1, len(selections) + 1)
        ):
            raise ValueError("selection ranks must be contiguous and match result order")
        if len({selection.shot_id for selection in selections}) != len(selections):
            raise ValueError("selected shot ids must be unique")
        if len({selection.selection_reason for selection in selections}) != len(selections):
            raise ValueError("selection reasons must be unique")
        if any(selection.rule_version is not self.rule_version for selection in selections):
            raise ValueError("every selection must use the result rule version")

        category_counts = {
            category: sum(
                selection.selection_reason.category == category for selection in selections
            )
            for category in ("severe", "repeated", "positive")
        }
        if (
            category_counts["severe"] > 2
            or category_counts["repeated"] > 2
            or category_counts["positive"] > 1
        ):
            raise ValueError("selection category quota exceeded")
        return self
