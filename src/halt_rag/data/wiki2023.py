"""Build the collection from the Wikipedia dump of 1 Nov 2023
(Hugging Face `wikimedia/wikipedia`, config `20231101.en`).

Why this is fast: the dump is stored as Parquet files. Parquet keeps each
column separately and records how many rows each file has, so we can
  * count all articles without reading any text,
  * pick random background rows by number,
  * read the text of only the rows we need.
KILT had to be streamed record by record (about 9 hours).

The dump has no Wikidata IDs, so matching works like this:
  Wikidata ID (from PopQA s_uri)
    -> English Wikipedia title   (Wikidata API, sitelinks)
    -> Wikipedia page ID         (Wikipedia API; follows redirects)
    -> the row in the dump with that page ID (the dump's `id` column)
Page IDs do not change when a page is renamed, which is why this is
more reliable than matching titles. Title matching is the backup.

Everything that touches the internet takes a `get` function, so the
logic can be tested offline.
"""
from __future__ import annotations

import time

import numpy as np

from .passages import clean_paragraphs
from .popqa import normalise_title

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "HALT-RAG student research project (https://github.com/Shaithra2107/HALT-RAG)"


def _chunks(xs, n):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def make_get(session=None, pause: float = 0.1, retries: int = 3):
    """A polite GET for the Wikimedia APIs (User-Agent set, small pause, retries)."""
    import requests
    s = session or requests.Session()
    s.headers["User-Agent"] = USER_AGENT

    def get(url, params):
        for attempt in range(retries):
            try:
                r = s.get(url, params={**params, "format": "json"}, timeout=30)
                r.raise_for_status()
                time.sleep(pause)
                return r.json()
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    return get


def qids_to_titles(qids, get) -> dict[str, str]:
    """Wikidata ID -> English Wikipedia title (50 IDs per request)."""
    out = {}
    for batch in _chunks(sorted(set(q for q in qids if q)), 50):
        data = get(WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(batch),
                                  "props": "sitelinks", "sitefilter": "enwiki"})
        for qid, ent in (data.get("entities") or {}).items():
            link = (ent.get("sitelinks") or {}).get("enwiki")
            if link and link.get("title"):
                out[qid] = link["title"]
    return out


def titles_to_pageids(titles, get) -> dict[str, int]:
    """Wikipedia title -> page ID, following redirects (50 titles per request).
    The result is keyed by the title as it was given."""
    out = {}
    for batch in _chunks(sorted(set(t for t in titles if t)), 50):
        data = get(WIKIPEDIA_API, {"action": "query", "titles": "|".join(batch),
                                   "redirects": 1, "formatversion": 2})
        q = data.get("query") or {}
        step = {}  # given title -> final title (normalised + redirects)
        for m in q.get("normalized", []):
            step[m["from"]] = m["to"]
        redirects = {m["from"]: m["to"] for m in q.get("redirects", [])}
        by_title = {p["title"]: p.get("pageid") for p in q.get("pages", [])
                    if not p.get("missing") and p.get("pageid")}
        for t in batch:
            final = step.get(t, t)
            final = redirects.get(final, final)
            if by_title.get(final):
                out[t] = int(by_title[final])
    return out


def split_text(text: str, min_chars: int = 50) -> list[str]:
    """The dump stores an article as plain text with one paragraph per line.
    Section headings are short lines, so the length filter drops them."""
    return clean_paragraphs(str(text or "").split("\n"), min_chars)


def plan_background(rows_per_file: list[int], n: int, seed: int,
                    exclude: set[tuple[int, int]] = frozenset(), oversample: float = 1.3):
    """Pick about n*oversample random (file, row) positions over the whole dump,
    uniformly, skipping excluded positions. Returns {file_index: sorted rows}
    and the order in which picks should be kept (so the first n valid ones win)."""
    rng = np.random.default_rng(seed)
    total = int(sum(rows_per_file))
    want = min(int(n * oversample) + len(exclude), total)
    picks = rng.choice(total, size=want, replace=False)
    starts = np.cumsum([0] + list(rows_per_file[:-1]))
    plan, order = {}, []
    for g in picks:                       # `picks` is already in random order
        f = int(np.searchsorted(starts, g, side="right") - 1)
        r = int(g - starts[f])
        if (f, r) in exclude:
            continue
        plan.setdefault(f, []).append(r)
        order.append((f, r))
    return {f: sorted(rs) for f, rs in plan.items()}, order


def build_targets(audit, get, qid_col="qid", title_col="wiki_title"):
    """For each question: the key used downstream (normalised wiki_title, same as v5),
    the page ID to look for, and how that page ID was found."""
    qid_title = qids_to_titles(audit[qid_col].dropna().tolist(), get)
    titles = set(qid_title.values()) | set(audit[title_col].dropna().astype(str))
    pid = titles_to_pageids(titles, get)

    targets = {}   # key -> {"pageid", "method", "titles"}
    for _, r in audit.iterrows():
        key = normalise_title(r[title_col])
        if key in targets:
            continue
        cand_titles = []
        q = r.get(qid_col)
        if isinstance(q, str) and q in qid_title:
            cand_titles.append(qid_title[q])
        cand_titles.append(str(r[title_col]))
        page, method = None, None
        if isinstance(q, str) and q in qid_title and qid_title[q] in pid:
            page, method = pid[qid_title[q]], "qid"
        elif str(r[title_col]) in pid:
            page, method = pid[str(r[title_col])], "title"
        targets[key] = {"pageid": page, "method": method,
                        "titles": {normalise_title(t) for t in cand_titles}}
    return targets
