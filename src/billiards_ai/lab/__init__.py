"""Local, artifact-first algorithm research laboratory."""

from billiards_ai.lab.contracts import (
    AdjudicationStatus,
    BallFrameAnnotation,
    CalibrationReference,
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    EventAnnotation,
    EventType,
    RunManifest,
    ShotAnnotation,
    VideoAsset,
    Visibility,
)
from billiards_ai.lab.dataset import (
    DatasetIntegrityError,
    GateUseAuthorizer,
    freeze_dataset,
    load_frozen_dataset,
)

__all__ = [
    "AdjudicationStatus",
    "BallFrameAnnotation",
    "CalibrationReference",
    "DatasetIntegrityError",
    "DatasetManifest",
    "DatasetPurpose",
    "DatasetSplit",
    "EventAnnotation",
    "EventType",
    "GateUseAuthorizer",
    "RunManifest",
    "ShotAnnotation",
    "VideoAsset",
    "Visibility",
    "freeze_dataset",
    "load_frozen_dataset",
]
