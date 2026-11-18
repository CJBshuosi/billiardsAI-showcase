"""Executable compatibility contract between final Story 9.7 and Story 10.1."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from billiards_ai.edge.plan_admission import (
    AntiRollbackDecision,
    CanonicalPublishedPlanBundleCodec,
    PlanBundleVerifier,
    UntrustedPlanBundle,
)
from billiards_ai.shared.plan_publication.admission import admission_view
from billiards_ai.shared.plan_publication.contracts import DecodedPublishedPlanBundle

ROOT = Path(__file__).parents[1] / "fixtures" / "plan_publication" / "v2"


def test_story_10_1_formal_codec_consumes_story_9_7_final_bundle() -> None:
    vector = json.loads((ROOT / "positive.json").read_text())

    decoded = CanonicalPublishedPlanBundleCodec().decode(
        base64.b64decode(vector["envelope_base64"], validate=True)
    )
    view = admission_view(decoded)

    assert type(decoded) is DecodedPublishedPlanBundle
    assert decoded.bundle_content_sha256 == vector["bundle_content_sha256"]
    assert view.not_before == decoded.payload.publication_policy.not_before
    assert tuple(item.drill_key for item in view.drill_definitions) == view.planned_drill_keys


class Clock:
    def __init__(self, now) -> None:
        self._now = now

    def now(self):
        return self._now


class Keyring:
    def __init__(self, public_key) -> None:
        self._public_key = public_key

    def resolve(self, **identity):
        return self._public_key


class AllowAuthorization:
    def authorize(self, claim) -> bool:
        return True


class AllowCapabilities:
    def compatible(self, *, target_device_id, manifest) -> bool:
        return True


class AcceptSequence:
    def accept(self, claim):
        return AntiRollbackDecision.ACCEPTED


def test_story_10_1_verification_order_accepts_real_story_9_7_crypto_vector() -> None:
    vector = json.loads((ROOT / "positive.json").read_text())
    envelope = base64.b64decode(vector["envelope_base64"], validate=True)
    decoded = CanonicalPublishedPlanBundleCodec().decode(envelope)
    public_key = decoded.payload.publication_policy.signing_key
    verifier = PlanBundleVerifier(
        clock=Clock(decoded.payload.publication_policy.issued_at),
        keyring=Keyring(public_key),
        authorization=AllowAuthorization(),
        capabilities=AllowCapabilities(),
        anti_rollback=AcceptSequence(),
    )

    result = verifier.verify(UntrustedPlanBundle(envelope))

    assert result.accepted
    assert result.admission is not None
    assert result.admission.plan_id == decoded.payload.plan.plan_id
