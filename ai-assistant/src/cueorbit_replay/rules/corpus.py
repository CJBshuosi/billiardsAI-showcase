"""Numbered source extraction. No runtime crawling or inferred rule text."""

import hashlib
import re
import unicodedata
from pathlib import Path

from .contracts import RuleChunk, RuleSourceManifestV1


def normalize_text(text: str) -> str:
    if "\ufffd" in text or any(ord(c) < 32 and c not in "\n\r\t" for c in text):
        raise ValueError("GARBLED_SOURCE")
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    ).strip()


def extract_chunks(
    pages: list[str], manifest: RuleSourceManifestV1, tokenizer
) -> tuple[RuleChunk, ...]:
    chunks = []
    section = None
    current = None
    lines = []
    seen = set()
    parents = {}

    def flush():
        if current is None or not lines:
            return
        rule, heading = current
        if len(lines) == 1 and lines[0][0] == heading:
            return
        full = ""
        locations = []
        for value, page in lines:
            if full:
                full += " "
            left = len(full)
            full += value
            locations.append((left, len(full), page))
        spans = [
            (m.start(), m.end())
            for m in re.finditer(r".+?(?:[.!?;](?=\s|$)|$)", full)
            if m.group().strip()
        ]
        start = 0
        part = 0
        while start < len(spans):
            end = start
            while end < len(spans):
                candidate = full[spans[start][0] : spans[end][1]].strip()
                if (
                    tokenizer.count(candidate) > 400
                    or tokenizer.count("passage: " + heading + " " + candidate) > 512
                ):
                    break
                end += 1
            if end == start:
                raise ValueError("SENTENCE_EXCEEDS_TOKEN_BUDGET")
            left, right = spans[start][0], spans[end - 1][1]
            body = full[left:right].strip()
            page_ids = [page for a, b, page in locations if a < right and b > left]
            digest = hashlib.sha256(body.encode()).hexdigest()
            identity = "|".join(
                (
                    manifest.source_id,
                    manifest.ruleset_id,
                    manifest.effective_date,
                    rule,
                    str(part),
                    digest,
                )
            )
            chunks.append(
                RuleChunk(
                    chunk_id="rule-"
                    + hashlib.sha256(identity.encode()).hexdigest()[:48],
                    source_id=manifest.source_id,
                    ruleset_id=manifest.ruleset_id,
                    version=manifest.effective_date,
                    rule_number=rule,
                    heading=heading,
                    page_start=min(page_ids),
                    page_end=max(page_ids),
                    text=body,
                    text_sha256=digest,
                    embedding_model=manifest.embedding_model,
                    embedding_revision=manifest.embedding_revision,
                    precedence="heyball_specific"
                    if rule.startswith("II.")
                    else "appendix_general",
                    data_kind=manifest.data_kind,
                )
            )
            part += 1
            if end == len(spans):
                break
            overlap = end
            while (
                overlap > start + 1
                and tokenizer.count(full[spans[overlap - 1][0] : spans[end - 1][1]])
                <= 40
            ):
                overlap -= 1
            start = overlap

    for page, raw in enumerate(pages, 1):
        for raw_line in raw.splitlines():
            line = normalize_text(raw_line)
            if not line:
                continue
            chapter = re.match(r"^CHAPTER\s+(III|II|I)\b", line, re.IGNORECASE)
            appendix = re.fullmatch(r"Appendix\s+A", line, re.IGNORECASE)
            if chapter or appendix:
                flush()
                lines = []
                current = None
                section = "A" if appendix else chapter.group(1).upper()
                continue
            if section not in manifest.scope_sections:
                continue
            if (
                line.startswith(("RULES OF PLAY", "Effective 2025-08-16", "Page "))
                or "World Pool-Billiard Association, 2025" in line
            ):
                continue
            match = re.match(r"^(\d+(?:\.\d+)+|\d+\.)\s+(.+)$", line)
            if match:
                flush()
                lines = []
                number = match.group(1).rstrip(".")
                rule = section + "." + number
                if rule in seen:
                    raise ValueError("DUPLICATE_RULE_NUMBER")
                seen.add(rule)
                content = match.group(2)
                if "." not in number:
                    parents[(section, number)] = content
                heading = parents.get((section, number.split(".")[0]), content)
                current = (rule, heading)
                lines.append((content, page))
            elif current is not None:
                lines.append((line, page))
    flush()
    if not chunks:
        raise ValueError("NO_IN_SCOPE_RULES")
    return tuple(chunks)


def extract_pdf(
    path: Path, manifest: RuleSourceManifestV1, tokenizer
) -> tuple[RuleChunk, ...]:
    from pypdf import PdfReader

    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest.pdf_sha256:
        raise ValueError("SOURCE_HASH_MISMATCH")
    reader = PdfReader(path)
    if manifest.data_kind == "official" and len(reader.pages) != 21:
        raise ValueError("SOURCE_PAGE_COUNT")
    return extract_chunks([p.extract_text() for p in reader.pages], manifest, tokenizer)
