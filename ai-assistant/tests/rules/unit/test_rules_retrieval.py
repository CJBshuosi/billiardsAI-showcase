from cueorbit_replay.rules.evaluation import retrieval_metrics


def test_recall_uses_all_relevant_chunks_and_mrr_first_relevant_rank():
    rows = [
        {
            "language": "en",
            "relevant_chunk_ids": ["a", "b"],
            "retrieved_chunk_ids": ["b", "x"],
        },
        {
            "language": "en",
            "relevant_chunk_ids": ["c"],
            "retrieved_chunk_ids": ["x", "c"],
        },
        {"language": "zh", "relevant_chunk_ids": [], "retrieved_chunk_ids": ["x"]},
    ]
    result = retrieval_metrics(rows)
    assert result["overall"]["recall_at_5"] == {
        "numerator": 2,
        "denominator": 3,
        "value": 2 / 3,
    }
    assert result["overall"]["hit_at_5"] == {
        "numerator": 2,
        "denominator": 2,
        "value": 1.0,
    }
    assert result["overall"]["mrr_at_5"] == {
        "numerator": 1.5,
        "denominator": 2,
        "value": 0.75,
    }
    assert result["by_language"]["zh"]["recall_at_5"]["value"] is None
    assert result["by_language"]["zh"]["answerable_queries"] == 0
    assert result["overall"]["excluded_queries"] == 1
