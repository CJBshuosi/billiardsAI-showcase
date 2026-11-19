from .contracts import ToolResult


def supported_refs(refs: tuple[str, ...], cards: tuple[ToolResult, ...]) -> bool:
    available = {
        ref
        for card in cards
        for ref in card.evidence_refs
        if card.status in ("ok", "partial")
    }
    return set(refs) <= available
