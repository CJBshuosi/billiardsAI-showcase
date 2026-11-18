from billiards_ai.lab.adapters.identity import HierarchicalIdentityAdapter


class FakeIdentityBackend:
    version = "synthetic-identity/1"

    def __init__(self, exact_scores: dict[str, float], group_scores: dict[str, float]) -> None:
        self.exact_scores = exact_scores
        self.group_scores = group_scores

    def predict(self, crop: object) -> dict:
        assert crop == "synthetic-crop"
        return {
            "exact_scores": self.exact_scores,
            "group_scores": self.group_scores,
            "visibility_scores": {"visible": 0.9, "occluded": 0.1},
        }


def test_low_evidence_returns_unknown_instead_of_highest_softmax_class() -> None:
    adapter = HierarchicalIdentityAdapter(
        FakeIdentityBackend(
            exact_scores={"8": 0.54, "3": 0.46},
            group_scores={"eight": 0.54, "solid": 0.46},
        ),
        exact_confidence_min=0.80,
        exact_margin_min=0.20,
    )

    result = adapter.resolve_identities(({"crop": "synthetic-crop", "track_id": "t1"},))[0]

    assert result["exact_identity"] is None
    assert result["group"] == "unknown"
    assert result["unknown_reason"] == "low_exact_confidence"
    assert result["forced_label"] is False


def test_exact_and_group_outputs_must_be_consistent() -> None:
    adapter = HierarchicalIdentityAdapter(
        FakeIdentityBackend(
            exact_scores={"10": 0.95, "2": 0.05},
            group_scores={"solid": 0.92, "stripe": 0.08},
        )
    )

    result = adapter.resolve_identities(({"crop": "synthetic-crop", "track_id": "t1"},))[0]

    assert result["exact_identity"] is None
    assert result["group"] == "solid"
    assert result["unknown_reason"] == "hierarchy_conflict"


def test_confident_group_is_retained_when_exact_identity_is_unresolved() -> None:
    adapter = HierarchicalIdentityAdapter(
        FakeIdentityBackend(
            exact_scores={"unknown": 0.55, "10": 0.45},
            group_scores={"stripe": 0.92, "solid": 0.08},
        )
    )

    result = adapter.resolve_identities(({"crop": "synthetic-crop", "track_id": "t1"},))[0]

    assert result["exact_identity"] is None
    assert result["group"] == "stripe"
    assert result["unknown_reason"] == "exact_unresolved"
