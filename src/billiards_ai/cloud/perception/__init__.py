"""Classic-CV, calibration-gated ball localization for Story 2.5."""

from billiards_ai.cloud.perception.adjudicator import (
    AdjudicationPolicy,
    AmbiguityAdjudicator,
    DisabledAdjudicationProvider,
    ProviderAdjudicationError,
)
from billiards_ai.cloud.perception.detector import ClassicCvBallDetector
from billiards_ai.cloud.perception.errors import DetectionError, DetectionInputError
from billiards_ai.cloud.perception.finalizer import assess_session_quality, finalize_shot
from billiards_ai.cloud.perception.observer import observe_shot
from billiards_ai.cloud.perception.service import BallLocatorService

__all__ = [
    "BallLocatorService",
    "AdjudicationPolicy",
    "AmbiguityAdjudicator",
    "ClassicCvBallDetector",
    "DetectionError",
    "DetectionInputError",
    "DisabledAdjudicationProvider",
    "ProviderAdjudicationError",
    "assess_session_quality",
    "finalize_shot",
    "observe_shot",
]
