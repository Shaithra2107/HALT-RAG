"""Offline tests for the 2023-dump helpers (the APIs are faked)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halt_rag.data.wiki2023 import (build_targets, plan_background, qids_to_titles,
                                    split_text, titles_to_pageids)


def fake_get(url, params):
    if "wikidata" in url:
        ents = {"Q90": {"sitelinks": {"enwiki": {"title": "Paris"}}},
                "Q1": {"sitelinks": {}}}                       # no English page
        return {"entities": {q: ents.get(q, {"missing": ""}) for q in params["ids"].split("|")}}
    # Wikipedia: "Old Name" redirects to "New Name"; "paris" is normalised to "Paris"
    return {"query": {
        "normalized": [{"from": "paris", "to": "Paris"}],
        "redirects": [{"from": "Old Name", "to": "New Name"}],
        "pages": [{"title": "Paris", "pageid": 22989},
                  {"title": "New Name", "pageid": 5},
                  {"title": "Gone", "missing": True}]}}


def test_qids_to_titles():
    assert qids_to_titles(["Q90", "Q1", None], fake_get) == {"Q90": "Paris"}


def test_titles_follow_normalising_and_redirects():
    got = titles_to_pageids(["paris", "Old Name", "Gone"], fake_get)
    assert got == {"paris": 22989, "Old Name": 5}


def test_build_targets_prefers_qid():
    audit = pd.DataFrame({"wiki_title": ["Paris (city)", "Old Name", "Gone"],
                          "qid": ["Q90", None, "Q1"]})
    t = build_targets(audit, fake_get)
    assert t["paris (city)"]["pageid"] == 22989 and t["paris (city)"]["method"] == "qid"
    assert t["old name"]["pageid"] == 5 and t["old name"]["method"] == "title"
    assert t["gone"]["pageid"] is None


def test_split_text_drops_headings():
    text = "History\n" + "a" * 60 + "\n\nEarly life\n" + "b" * 60
    assert split_text(text) == ["a" * 60, "b" * 60]


def test_plan_background_is_uniform_and_skips_targets():
    plan, order = plan_background([100, 50, 150], n=50, seed=42, exclude={(0, 3)})
    assert (0, 3) not in order
    assert len(order) == len(set(order)) >= 50
    assert all(0 <= r < [100, 50, 150][f] for f, rs in plan.items() for r in rs)
    plan2, order2 = plan_background([100, 50, 150], n=50, seed=42, exclude={(0, 3)})
    assert order == order2                      # same seed -> same sample
