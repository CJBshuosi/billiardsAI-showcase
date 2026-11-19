"""Owned startup/readiness/shutdown for the optional local rules feature."""

import importlib.util
import json

from ..fixtures import ROOT
from .contracts import RulesContext, RuleSourceManifestV1
from .model import OfflineReviewModel, RulesDeepSeekModel
from .worker import EmbeddingWorker


class RulesRuntime:
    review_enabled = True

    def __init__(self, settings):
        self.settings = settings
        self.strategy = settings.rules_retrieval
        self.encoder = None
        self.repository = None
        self.context = None
        self._ready = False
        self.reason = "STARTING"

    @property
    def ready(self):
        return self._ready and self.encoder is not None and self.encoder.ready

    async def start(self):
        if any(
            importlib.util.find_spec(name) is None
            for name in ("fastembed", "onnxruntime", "pgvector")
        ):
            self.reason = "DEPENDENCY_MISSING"
            return
        stage = "CORPUS_METADATA_INVALID"
        try:
            if self.strategy not in ("keyword", "vector", "vector_glossary_v1"):
                raise ValueError("INVALID_STRATEGY")
            metadata = json.loads((ROOT / "data/rules/chunk-manifest.json").read_text())
            manifest = RuleSourceManifestV1.model_validate(metadata["source"])
            if manifest.data_kind != "official":
                raise ValueError("OFFICIAL_CORPUS_REQUIRED")
            expected = RulesContext.model_validate(metadata["active_context"])
            from .annotations import load_annotations, validate_registry_metadata

            validate_registry_metadata(load_annotations(), metadata)
            stage = "MODEL_UNAVAILABLE"
            self.encoder = EmbeddingWorker(
                self.settings.rules_encoder_dir
                or ROOT / "work/models/multilingual-e5-small"
            )
            await self.encoder.start()
            stage = "INDEX_UNAVAILABLE"
            from .retrieval import RuleRepository

            self.repository = RuleRepository(
                self.settings.rules_database_url
                or "postgresql+asyncpg://replay@127.0.0.1:54331/cueorbit_replay_m1_m2"
            )
            import asyncio

            actual = await asyncio.wait_for(self.repository.context_for(manifest), 2)
            if actual != expected:
                raise ValueError("INDEX_IDENTITY_MISMATCH")
            self.context = actual
            self._ready = True
            self.reason = "READY"
        except Exception:
            self.reason = stage
            await self.aclose()

    def capability(self):
        ready = self.ready
        return {
            "enabled": True,
            "ready": ready,
            "reason": "READY"
            if ready
            else "EMBEDDING_UNAVAILABLE"
            if self._ready
            else self.reason,
            "rulesets": [
                {
                    "ruleset_id": self.context.ruleset_id,
                    "version": self.context.version,
                    "label": "\u4e2d\u5f0f\u516b\u7403 \u00b7 WPA Heyball \u00b7 "
                    + self.context.version,
                }
            ]
            if self.context
            else [],
        }

    def model_for(self, mode):
        return (
            OfflineReviewModel()
            if mode == "offline"
            else RulesDeepSeekModel(self.settings.api_key, self.settings.model_name)
        )

    async def aclose(self):
        self._ready = False
        if self.encoder is not None:
            await self.encoder.aclose()
        if self.repository is not None:
            await self.repository.close()
