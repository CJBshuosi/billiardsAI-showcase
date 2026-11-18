"""Public protocol implemented by every pluggable vision backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from billiards_ai.lab.vision.contracts import (
    VisionFrameInput,
    VisionFrameObservation,
    VisionPackageIdentity,
)


class VisionBackend(Protocol):
    @property
    def package_identity(self) -> VisionPackageIdentity: ...

    def infer(self, frame: VisionFrameInput) -> VisionFrameObservation: ...


class BatchVisionBackend(VisionBackend, Protocol):
    def infer_batch(
        self,
        frames: Sequence[VisionFrameInput],
    ) -> tuple[VisionFrameObservation, ...]: ...
