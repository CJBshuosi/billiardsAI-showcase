"""Reuse the existing model observation adapter at the stable-snapshot boundary."""

from types import SimpleNamespace

import pytest

from tests.unit.lab.field_assistance.test_snapshot_guidance import frame


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_snapshot_factory_loads_identity_weights_before_returning_ready_without_frames(
    tmp_path, monkeypatch, fail_preparation
):
    from billiards_ai.lab import mainline1_runtime_support as runtime
    from billiards_ai.lab.field_assistance.snapshot_models import ModelSnapshotObserverFactory

    calls = []
    failure = RuntimeError("controlled identity preparation failure")
    identity = object()
    vision = SimpleNamespace(
        prepare_execution=lambda **kwargs: calls.append(("detector_execution", kwargs))
    )

    class Features:
        prepared = False

        def __init__(self, **kwargs):
            calls.append("features_created")

        def prepare(self):
            calls.append("identity_weights_loaded")
            if fail_preparation:
                raise failure
            self.prepared = True

        def embed(self, *_):
            pytest.fail("cold preparation must not receive crops")

    def classifier(*args, **kwargs):
        assert kwargs["feature_backend"].prepared
        calls.append("classifier_created")
        return identity

    monkeypatch.setattr(runtime, "load_development_vision_backend", lambda *a, **kw: vision)
    monkeypatch.setattr(runtime, "Dinov2IdentityFeatureBackend", Features)
    monkeypatch.setattr(runtime, "load_development_group_classifier", classifier)
    factory = ModelSnapshotObserverFactory(tmp_path, tmp_path, tmp_path)
    if fail_preparation:
        with pytest.raises(RuntimeError) as raised:
            factory()
        assert raised.value is failure
        assert calls == ["features_created", "identity_weights_loaded"]
    else:
        observer = factory()
        assert observer.vision is vision and observer.identity is identity
        assert calls == [
            "features_created",
            "identity_weights_loaded",
            "classifier_created",
            ("detector_execution", {"width_px": 1920, "height_px": 1080}),
        ]


def test_explicit_identity_preparation_reuses_original_loader_without_inference(
    tmp_path, monkeypatch
):
    import sys

    from billiards_ai.lab.mainline1_identity_backends import Dinov2IdentityFeatureBackend

    calls = []

    class Model:
        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            calls.append("eval")

        def __call__(self, *_):
            pytest.fail("model preparation must not infer")

    model, processor = Model(), object()

    def load_processor(*args, **kwargs):
        calls.append(("processor", kwargs))
        return processor

    def load_model(*args, **kwargs):
        calls.append(("weights", kwargs))
        return model

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=SimpleNamespace(from_pretrained=load_processor),
            AutoModel=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    backend = object.__new__(Dinov2IdentityFeatureBackend)
    backend.asset_root, backend.device = tmp_path, "cpu"
    backend._processor = backend._model = backend._torch = None
    backend.prepare()
    backend.prepare()
    assert backend._model is model and backend._processor is processor
    assert calls == [
        ("processor", {"local_files_only": True}),
        ("weights", {"local_files_only": True, "use_safetensors": True}),
        ("device", "cpu"),
        "eval",
    ]


def test_model_snapshot_uses_one_decoded_frame_and_existing_adapter(monkeypatch):
    from billiards_ai.lab.field_assistance import snapshot_models

    calls = []
    expected = SimpleNamespace(frame_id=5)

    def collect(**kwargs):
        calls.append(kwargs)
        return (expected,)

    monkeypatch.setattr(snapshot_models, "collect_observation_frames", collect)
    observer = snapshot_models.ModelSnapshotObserver("vision", "identity")
    assert observer.observe(frame(5), "calibration") is expected
    assert len(calls) == 1
    assert len(calls[0]["frames"]) == 1
    assert calls[0]["frames"][0].image.shape == (240, 320, 3)
    assert calls[0]["frames"][0].frame_id == 5
    assert calls[0]["vision_backend"] == "vision"
    assert calls[0]["identity_classifier"] == "identity"
