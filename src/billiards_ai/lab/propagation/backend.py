"""Port implemented by bounded video-mask propagation teachers."""

from __future__ import annotations

from typing import Protocol

from billiards_ai.lab.propagation.contracts import (
    InstancePrompt,
    PropagationResult,
    PropagationWindow,
)


class PropagationBackend(Protocol):
    backend_id: str
    backend_version: str

    def propagate(
        self,
        *,
        window: PropagationWindow,
        prompts: tuple[InstancePrompt, ...],
    ) -> PropagationResult: ...
