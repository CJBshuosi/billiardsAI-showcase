"""Explicit versioned bilingual terms; never modify the raw baselines."""

import json
from functools import lru_cache

from ..fixtures import ROOT


@lru_cache(maxsize=1)
def glossary():
    return json.loads((ROOT / "data/rules/query-glossary.json").read_text())


def expand_query(question: str) -> str:
    terms = list(
        dict.fromkeys(
            value for key, value in glossary()["terms"].items() if key in question
        )
    )
    return " / ".join(terms) + ". " + question if terms else question
