"""Quick tests: run with  python -m pytest tests  (from the repo root)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halt_rag.data.kilt_scan import Targets, report, scan
from halt_rag.data.passages import dedupe, merge_into_passages
from halt_rag.data.popqa import parse_answers, wikidata_qid
from halt_rag.eval.metrics import auroc_ci
from halt_rag.signals.embedding import lift, lift_closed_form


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_wikidata_qid():
    assert wikidata_qid("http://www.wikidata.org/entity/Q90") == "Q90"
    assert wikidata_qid(None) is None
    assert wikidata_qid("no id here") is None


def test_parse_answers():
    assert parse_answers('["Paris", "City of Light"]') == ["Paris", "City of Light"]
    assert parse_answers(["a", " ", "nan"]) == ["a"]


def test_merge_joins_short_and_cuts_long():
    short = ["one two three"] * 5                       # 15 words in total
    assert merge_into_passages(short, max_words=100, min_chars=0) == [" ".join(short)]
    long = [" ".join(["w"] * 250)]
    out = merge_into_passages(long, max_words=100, min_chars=0)
    assert [len(p.split()) for p in out] == [100, 100, 50]


def test_dedupe_keeps_rows_aligned():
    p, t = dedupe(["A b", "a  B", "c"], ["t1", "t2", "t3"])
    assert p == ["A b", "c"] and t == ["t1", "t3"]


def test_lift_matches_v5_and_uses_eps():
    rng = np.random.default_rng(0)
    pool = unit(rng.normal(size=(200, 8)))
    q = unit(rng.normal(size=8))
    top = unit(q + 0.1 * rng.normal(size=(5, 8)))
    l, mu, sd = lift(q, top, pool)
    rs = pool @ q
    assert np.isclose(l, ((top @ q).mean() - rs.mean()) / rs.std())   # v5 formula
    # a pool with zero spread: v5 returned 0.0, now eps keeps it finite and positive
    flat = np.tile(pool[:1], (10, 1))
    l2, _, sd2 = lift(q, top, flat, eps=1e-6)
    assert sd2 == 0 and np.isfinite(l2)


def test_lift_excludes_own_passages():
    rng = np.random.default_rng(1)
    pool = unit(rng.normal(size=(50, 4)))
    q = unit(rng.normal(size=4))
    pool_ids = np.arange(100, 150)
    _, mu_all, _ = lift(q, pool[:5], pool)
    _, mu_ex, _ = lift(q, pool[:5], pool, pool_ids, ret_ids=[100, 101])
    assert np.isclose(mu_ex, (pool[2:] @ q).mean()) and not np.isclose(mu_all, mu_ex)


def test_closed_form_equals_loop():
    rng = np.random.default_rng(2)
    pool = unit(rng.normal(size=(500, 6)))
    q = unit(rng.normal(size=6))
    top = pool[:5]
    l_loop, _, _ = lift(q, top, pool)
    cov = np.cov(pool.T, bias=True)
    l_cf, _, _ = lift_closed_form(q, top, pool.mean(0), cov)
    assert np.isclose(l_loop, l_cf)


def test_scan_matches_qid_first_then_title():
    targets = Targets.from_frame(pd.DataFrame({
        "wiki_title": ["Paris", "Old Name"], "qid": ["Q90", None]}))
    para = "x" * 60
    recs = [
        {"wikipedia_id": "1", "wikipedia_title": "Paris, France",          # renamed page
         "wikidata_info": {"wikidata_id": "Q90"}, "text": {"paragraph": [para]}},
        {"wikipedia_id": "2", "wikipedia_title": "Old_Name",
         "wikidata_info": None, "text": {"paragraph": [para]}},
        {"wikipedia_id": "3", "wikipedia_title": "Other",
         "wikidata_info": {"wikidata_id": "Q1"}, "text": {"paragraph": [para]}},
        {"wikipedia_id": "3", "wikipedia_title": "Other",                  # duplicate record
         "wikidata_info": {"wikidata_id": "Q1"}, "text": {"paragraph": ["short"]}},
    ]
    st = scan(recs, targets, reservoir_size=10, min_para_chars=50,
              max_para_per_background=40, seed=42)
    r = report(st, targets)
    assert st.match_method == {"paris": "qid", "old name": "title"}
    assert r["duplicate_records"] == 1 and r["records_without_usable_text"] == 1
    assert r["targets_missing"] == 0 and r["background_pages"] == 1


def test_auroc_ci():
    y = np.array([0, 0, 1, 1] * 25)
    s = y + np.random.default_rng(0).normal(0, 0.5, size=100)
    auc, lo, hi = auroc_ci(y, s, n_boot=200)
    assert lo <= auc <= hi and auc > 0.8
