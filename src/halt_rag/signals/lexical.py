"""Keyword coverage and clarity: the two word-count signals (same as v5 cell 6).

Clarity (Cronen-Townsend et al., 2002) = KL divergence between the word mix
of the retrieved passages and the word mix of the WHOLE corpus.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("""a an the is are was were be been being of in on at to for with by from as and or but
if then than that this these those what which who whom whose when where why how do does did done
it its his her their there here not no nor so such own same too very can will just should now
you your i me my we our they them he she him""".split())


def toks(s) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def content(words) -> list[str]:
    return [w for w in words if w not in STOP and len(w) > 2]


class CorpusLM:
    """Word counts over every passage in the corpus."""

    def __init__(self, passages):
        self.counts = Counter()
        for t in passages:
            self.counts.update(content(toks(t)))
        self.total = sum(self.counts.values())

    def prob(self, w: str) -> float:
        return self.counts.get(w, 0) / self.total if self.total else 0.0


def kw_coverage(question: str, passages) -> float:
    """Share of the question's content words that appear in the passages."""
    kws = set(content(toks(question)))
    if not kws:
        return 0.0
    return len(kws & set(toks(" ".join(passages)))) / len(kws)


def clarity(passages, lm: CorpusLM, smooth: float = 1e-9) -> float:
    c = Counter(content(toks(" ".join(passages))))
    n = sum(c.values())
    if n == 0:
        return 0.0
    return float(sum((f / n) * math.log((f / n) / (lm.prob(w) or smooth))
                     for w, f in c.items()))
