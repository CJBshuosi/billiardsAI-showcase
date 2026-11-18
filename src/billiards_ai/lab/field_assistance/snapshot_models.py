"""Spawn-safe model construction; reuse the existing single-frame observation adapter.

Construction/inference happens inside the cancellable snapshot child, never the
capture or recording thread. Missing model assets fail explicitly; no replacement
observer is selected. This adapter does not claim field model acceptance.
"""

from dataclasses import dataclass
from pathlib import Path

from billiards_ai.lab.shot_local.observations import (
    DecodedTrackingFrame,
    collect_observation_frames,
)


@dataclass
class ModelSnapshotObserver:
    vision: object
    identity: object

    def observe(self, frame, calibration):
        return collect_observation_frames(
            frames=(
                DecodedTrackingFrame(
                    frame.capture_frame_id, frame.timestamp_ms, frame.decode_bgr()
                ),
            ),
            vision_backend=self.vision,
            identity_classifier=self.identity,
            calibration=calibration,
        )[0]


@dataclass(frozen=True)
class ModelSnapshotObserverFactory:
    detector_training_run: Path
    identity_training_run: Path
    dinov2_assets: Path
    device: str = "cpu"
    batch_size: int = 8

    def __call__(self):
        from billiards_ai.lab import mainline1_runtime_support as runtime

        vision = runtime.load_development_vision_backend(
            self.detector_training_run,
            model_factory=lambda path: runtime._detector(path, device=self.device),
        )
        features = runtime.Dinov2IdentityFeatureBackend(
            asset_root=self.dinov2_assets, device=self.device, batch_size=self.batch_size
        )
        features.prepare()
        identity = runtime.load_development_group_classifier(
            self.identity_training_run,
            feature_backend=features,
            evidence_backend=runtime.OpenCvIdentityVisualEvidenceBackend(),
            session_factory=runtime._session,
        )
        vision.prepare_execution(width_px=1920, height_px=1080)
        return ModelSnapshotObserver(vision, identity)
