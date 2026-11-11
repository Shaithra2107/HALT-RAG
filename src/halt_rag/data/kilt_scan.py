"""One pass over KILT Wikipedia: find the subject pages and draw the random background.

Changes from HALT_RAG_v5_scan.ipynb
1. Match by Wikidata ID first (PopQA s_uri <-> KILT wikidata_info.wikidata_id),
   and use the title only as a backup. v5 matched by title only.
2. Count every record's wikipedia_id, so we can see whether the 6.3M records
   (vs 5,903,530 pages) come from duplicates.
3. Record HOW each page was matched (qid / title) for the write-up.

The loop itself (reservoir sampling = Algorithm R, checkpoints) is the same as v5.
The code here has no Colab or Drive parts, so it can be tested with fake records.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, field

import numpy as np

from .passages import clean_paragraphs
from .popqa import normalise_title


@dataclass
class Targets:
    """What we are looking for. Keys are what a KILT record can match on."""
    by_qid: dict[str, str]     # 'Q90' -> target key
    by_title: dict[str, str]   # normalised title -> target key

    @classmethod
    def from_frame(cls, df, key_col="wiki_title", qid_col="qid", title_col="wiki_title"):
        by_qid, by_title = {}, {}
        for _, r in df.iterrows():
            key = normalise_title(r[key_col])
            q = r.get(qid_col)
            if isinstance(q, str) and q:
                by_qid[q] = key
            by_title[normalise_title(r[title_col])] = key
        return cls(by_qid, by_title)

    @property
    def keys(self) -> set[str]:
        return set(self.by_qid.values()) | set(self.by_title.values())


@dataclass
class ScanState:
    """Everything needed to resume a scan. Pickle this for checkpoints."""
    kilt_pages: dict = field(default_factory=dict)   # target key -> paragraphs
    match_method: dict = field(default_factory=dict) # target key -> 'qid' | 'title'
    reservoir: list = field(default_factory=list)
    scanned: int = 0
    n_eligible: int = 0
    n_empty: int = 0          # records with no usable paragraph
    n_no_qid: int = 0         # records with no Wikidata ID
    wiki_ids: array = field(default_factory=lambda: array("q"))
    rng_state: dict | None = None


def _record_qid(rec) -> str | None:
    info = rec.get("wikidata_info") or {}
    q = info.get("wikidata_id") if isinstance(info, dict) else None
    return q if isinstance(q, str) and q.startswith("Q") else None


def _record_paragraphs(rec) -> list:
    text = rec.get("text")
    return text.get("paragraph", []) if isinstance(text, dict) else []


def scan(records, targets: Targets, *, reservoir_size: int, min_para_chars: int,
         max_para_per_background: int, seed: int, state: ScanState | None = None,
         limit: int | None = None, checkpoint_every: int | None = None,
         on_checkpoint=None) -> ScanState:
    """Stream KILT records once. `records` is any iterable of KILT dicts."""
    st = state or ScanState()
    rng = np.random.default_rng(seed)
    if st.rng_state is not None:
        rng.bit_generator.state = st.rng_state

    for rec in records:
        if limit is not None and st.scanned >= limit:
            break
        st.scanned += 1

        try:
            st.wiki_ids.append(int(rec.get("wikipedia_id")))
        except (TypeError, ValueError):
            st.wiki_ids.append(-1)

        qid = _record_qid(rec)
        if qid is None:
            st.n_no_qid += 1

        paras = clean_paragraphs(_record_paragraphs(rec), min_para_chars)
        if not paras:
            st.n_empty += 1
            continue

        # 1) Wikidata ID  2) title backup (never overwrite a qid match)
        key, how = None, None
        if qid is not None and qid in targets.by_qid:
            key, how = targets.by_qid[qid], "qid"
        else:
            t = normalise_title(rec.get("wikipedia_title", ""))
            if t in targets.by_title:
                key, how = targets.by_title[t], "title"

        if key is not None:
            if how == "qid" or st.match_method.get(key) != "qid":
                st.kilt_pages[key] = paras
                st.match_method[key] = how
            continue

        # background page: Algorithm R keeps a uniform sample of all eligible pages
        st.n_eligible += 1
        item = {"title": rec.get("wikipedia_title", ""), "qid": qid,
                "paragraphs": paras[:max_para_per_background]}
        if len(st.reservoir) < reservoir_size:
            st.reservoir.append(item)
        else:
            j = int(rng.integers(0, st.n_eligible))
            if j < reservoir_size:
                st.reservoir[j] = item

        if checkpoint_every and on_checkpoint and st.scanned % checkpoint_every == 0:
            st.rng_state = rng.bit_generator.state
            on_checkpoint(st)

    st.rng_state = rng.bit_generator.state
    return st


def report(st: ScanState, targets: Targets) -> dict:
    """Numbers for the write-up and for Dr. Kasuni."""
    ids = np.frombuffer(st.wiki_ids, dtype=np.int64) if len(st.wiki_ids) else np.array([], np.int64)
    valid = ids[ids >= 0]
    uniq = np.unique(valid)
    found = set(st.kilt_pages)
    methods = list(st.match_method.values())
    return {
        "records_scanned": st.scanned,
        "unique_wikipedia_ids": int(len(uniq)),
        "duplicate_records": int(len(valid) - len(uniq)),
        "records_without_id": int((ids < 0).sum()),
        "records_without_wikidata_id": st.n_no_qid,
        "records_without_usable_text": st.n_empty,
        "targets_total": len(targets.keys),
        "targets_found": len(found),
        "found_by_qid": methods.count("qid"),
        "found_by_title_only": methods.count("title"),
        "targets_missing": len(targets.keys - found),
        "background_pages": len(st.reservoir),
        "background_eligible": st.n_eligible,
    }
