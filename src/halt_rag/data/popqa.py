"""PopQA helpers: answers, Wikidata IDs, head/tail sampling."""
from __future__ import annotations

import ast
import re

import pandas as pd

_QID = re.compile(r"(Q\d+)\s*$")


def parse_answers(val) -> list[str]:
    """Turn PopQA's possible_answers (a list, or a string of a list) into a clean list."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple)):
        ans = val
    else:
        try:
            ans = ast.literal_eval(str(val))
        except Exception:
            ans = [str(val)]
        if isinstance(ans, str):
            ans = [ans]
    return [str(a).strip() for a in ans
            if str(a).strip() and str(a).strip().lower() != "nan"]


def wikidata_qid(uri) -> str | None:
    """'http://www.wikidata.org/entity/Q90' -> 'Q90'. Returns None if there is no ID."""
    if uri is None or (isinstance(uri, float) and pd.isna(uri)):
        return None
    m = _QID.search(str(uri).strip())
    return m.group(1) if m else None


def normalise_title(t) -> str:
    """Same title key the v5 notebooks used: lower case, underscores -> spaces."""
    return re.sub(r"_", " ", str(t).lower().strip())


def sample_bands(df: pd.DataFrame, n_per_band: int, tail_max_pop: int,
                 head_min_pop: int, seed: int) -> pd.DataFrame:
    """Pick n tail questions (s_pop < tail_max_pop) and n head questions
    (s_pop > head_min_pop), the same rule as the pilot notebook."""
    tail = df[df["s_pop"] < tail_max_pop]
    head = df[df["s_pop"] > head_min_pop]
    t = tail.sample(n=min(n_per_band, len(tail)), random_state=seed).assign(band="tail")
    h = head.sample(n=min(n_per_band, len(head)), random_state=seed).assign(band="head")
    out = pd.concat([t, h]).reset_index(drop=True)
    out["answers_list"] = out["possible_answers"].apply(parse_answers)
    out["qid"] = out["s_uri"].apply(wikidata_qid)
    return out
