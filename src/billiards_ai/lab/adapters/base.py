"""Narrow protocols that keep candidate code separate from scoring and reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

Observation = Mapping[str, Any]


class Detector(Protocol):
    def detect(self, frames: Sequence[Observation]) -> tuple[Observation, ...]: ...


class IdentityResolver(Protocol):
    def resolve_identities(self, detections: Sequence[Observation]) -> tuple[Observation, ...]: ...


class Tracker(Protocol):
    def track(self, observations: Sequence[Observation]) -> tuple[Observation, ...]: ...


class ShotSegmenter(Protocol):
    def segment_shots(self, tracks: Sequence[Observation]) -> tuple[Observation, ...]: ...


class EventObserver(Protocol):
    def observe_events(self, shots: Sequence[Observation]) -> tuple[Observation, ...]: ...


class Renderer(Protocol):
    name: str
    version: str

    def render(
        self,
        *,
        output_path: Path,
        overlay_frames: tuple[dict[str, Any], ...],
    ) -> None: ...
