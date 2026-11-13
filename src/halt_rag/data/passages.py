"""Turn Wikipedia paragraphs into passages.

Change from v5: v5 cut every paragraph on its own, so a 20-word paragraph
became a 20-word passage. Here short paragraphs from the same page are
joined until a passage reaches about `max_words`, and long paragraphs are
still cut into `max_words` pieces. Exact duplicates are removed.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# collapses consecutive spaces/tabs after NFKC converts non-breaking spaces
_WS = re.compile(r'[ \t]+')
# sentence boundary: .  !  ? followed by whitespace
_SENT_END = re.compile(r'(?<=[.!?])\s+')


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT_END.split(text.strip()) if s.strip()]


def clean_paragraphs(paras, min_chars: int = 50) -> list[str]:
    """Keep real text paragraphs of at least `min_chars` characters.
    Applies Unicode NFKC normalisation (converts non-breaking spaces, ligatures,
    full-width chars …) then collapses repeated whitespace."""
    cleaned = []
    for p in (paras or []):
        if not isinstance(p, str):
            continue
        p = unicodedata.normalize("NFKC", p)   # \xa0 -> space, ligatures, etc.
        p = _WS.sub(" ", p).strip()
        if len(p) > min_chars:
            cleaned.append(p)
    return cleaned


def merge_into_passages(paras, max_words: int = 100, min_chars: int = 30) -> list[str]:
    """Join consecutive paragraphs of one page into passages of up to `max_words` words.
    Cuts at sentence boundaries where possible; only breaks mid-sentence when a
    single sentence exceeds `max_words`."""
    out, buf = [], []

    def flush():
        if buf:
            text = " ".join(buf)
            if len(text) > min_chars:
                out.append(text)
            buf.clear()

    for para in paras:
        for sent in _sentences(str(para)):
            words = sent.split()
            if len(words) > max_words:          # single sentence too long: cut at word boundaries
                flush()
                for i in range(0, len(words), max_words):
                    buf.extend(words[i:i + max_words])
                    flush()
                continue
            if len(buf) + len(words) > max_words:
                flush()
            buf.extend(words)
    flush()
    return out


def _key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def dedupe(passages: list[str], *others: list) -> tuple:
    """Drop passages whose normalised text was already seen.
    Extra lists (titles, flags...) are filtered the same way, so rows stay aligned."""
    seen, keep = set(), []
    for i, p in enumerate(passages):
        k = _key(p)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    result = [[passages[i] for i in keep]]
    for lst in others:
        result.append([lst[i] for i in keep])
    return tuple(result) if others else result[0]
