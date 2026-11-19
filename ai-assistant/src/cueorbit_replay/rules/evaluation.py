"""Mechanical retrieval metrics; not semantic answer or model scores."""


def fraction(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def retrieval_metrics(rows):
    def summarize(items):
        relevant_count = found_count = hit_count = answerable = 0
        reciprocal = 0.0
        for row in items:
            relevant = set(row["relevant_chunk_ids"])
            if not relevant:
                continue
            answerable += 1
            retrieved = row["retrieved_chunk_ids"][:5]
            relevant_count += len(relevant)
            found_count += len(relevant.intersection(retrieved))
            rank = next(
                (i for i, value in enumerate(retrieved, 1) if value in relevant), None
            )
            if rank is not None:
                hit_count += 1
                reciprocal += 1 / rank
        return {
            "total_queries": len(items),
            "answerable_queries": answerable,
            "excluded_queries": len(items) - answerable,
            "recall_at_5": fraction(found_count, relevant_count),
            "hit_at_5": fraction(hit_count, answerable),
            "mrr_at_5": fraction(reciprocal, answerable),
        }

    return {
        "overall": summarize(rows),
        "by_language": {
            language: summarize([r for r in rows if r["language"] == language])
            for language in ("en", "zh")
        },
    }
