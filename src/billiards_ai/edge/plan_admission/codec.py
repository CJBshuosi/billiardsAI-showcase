"""Consumer adapter for the authoritative Story 9.7 publication codec."""

from billiards_ai.shared.plan_publication.codec import (
    CanonicalCodecError,
    decode_bundle,
    decode_bundle_v3,
)
from billiards_ai.shared.plan_publication.contracts import (
    DecodedPublishedPlanBundle,
    DecodedPublishedPlanBundleV3,
)


class CanonicalPublishedPlanBundleCodec:
    """Decode only exact canonical executable final-v3 publication envelopes."""

    def decode(
        self, envelope_bytes: bytes
    ) -> DecodedPublishedPlanBundleV3 | DecodedPublishedPlanBundle:
        try:
            return decode_bundle_v3(envelope_bytes)
        except CanonicalCodecError:
            return decode_bundle(envelope_bytes)


__all__ = ["CanonicalPublishedPlanBundleCodec"]
