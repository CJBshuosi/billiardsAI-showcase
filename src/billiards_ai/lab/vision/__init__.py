"""Stable boundary between vision models and the algorithm brain."""

from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameInput,
    VisionFrameObservation,
    VisionPackageIdentity,
    make_vision_frame_input,
)
from billiards_ai.lab.vision.package import (
    VerifiedVisionPackage,
    VisionArtifact,
    VisionModelPackageManifest,
    VisionPackageError,
    VisionPostprocess,
    VisionPreprocess,
    canonical_package_id,
    load_vision_package,
)
from billiards_ai.lab.vision.promotion import (
    BenchmarkIdentity,
    BenchmarkSuiteResult,
    RelativeMetricRule,
    VisionBenchmarkResult,
    VisionPromotionPolicy,
    VisionPromotionVerdict,
    evaluate_promotion,
)
from billiards_ai.lab.vision.registry import (
    ActiveVisionModel,
    VisionRegistry,
    VisionRegistryEntry,
    VisionRegistryError,
    activate_candidate,
    load_active_model,
    resolve_active_package,
    rollback_active,
)
from billiards_ai.lab.vision.rfdetr import (
    RfdetrVisionBackend,
    VisionInferenceError,
)

__all__ = [
    "ActiveVisionModel",
    "BallInstanceObservation",
    "BenchmarkIdentity",
    "BenchmarkSuiteResult",
    "PixelBox",
    "PixelPoint",
    "RelativeMetricRule",
    "RfdetrVisionBackend",
    "VerifiedVisionPackage",
    "VisionArtifact",
    "VisionBackend",
    "VisionBenchmarkResult",
    "VisionFrameInput",
    "VisionFrameObservation",
    "VisionInferenceError",
    "VisionModelPackageManifest",
    "VisionPackageError",
    "VisionPackageIdentity",
    "VisionPostprocess",
    "VisionPreprocess",
    "VisionPromotionPolicy",
    "VisionPromotionVerdict",
    "VisionRegistry",
    "VisionRegistryEntry",
    "VisionRegistryError",
    "activate_candidate",
    "canonical_package_id",
    "evaluate_promotion",
    "load_active_model",
    "load_vision_package",
    "make_vision_frame_input",
    "resolve_active_package",
    "rollback_active",
]
