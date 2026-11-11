"""Lift, relevance and consistency: the three signals built from vectors.

Lift(q) = (mean sim(q, top-k) - mu_R(q)) / max(sigma_R(q), eps)

R is ONE random sample of corpus passages (the "Lift pool"), drawn once
and shared by every question. A question's own retrieved passages are
removed from R before mu and sigma are taken.

Same maths as v5 cell 5. The only change: max(sigma, eps) replaces
"if sd > 0 else 0.0", so Lift never divides by ~0 and never jumps to 0.
"""
from __future__ import annotations

import numpy as np


def draw_lift_pool(n_corpus: int, pool_size: int, seed: int) -> np.ndarray:
    """Indices of the random corpus passages used as Lift's baseline."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_corpus, size=min(pool_size, n_corpus), replace=False)


def relevance(q_vec: np.ndarray, p_vecs: np.ndarray) -> float:
    """Mean similarity between the question and its top-k passages."""
    return float((p_vecs @ q_vec).mean())


def consistency(p_vecs: np.ndarray) -> float:
    """Mean similarity between every pair of retrieved passages."""
    if len(p_vecs) < 2:
        return 1.0
    g = p_vecs @ p_vecs.T
    iu = np.triu_indices(len(p_vecs), k=1)
    return float(g[iu].mean())


def lift(q_vec, p_vecs, pool_vecs, pool_ids=None, ret_ids=None, eps: float = 1e-6):
    """Returns (lift, mu_R, sigma_R) for one question.

    pool_ids / ret_ids are corpus indices; when both are given, retrieved
    passages that also sit in the pool are left out of mu and sigma.
    """
    rs = pool_vecs @ q_vec
    if pool_ids is not None and ret_ids is not None:
        keep = ~np.isin(pool_ids, np.asarray(ret_ids))
        rs = rs[keep]
    mu, sd = float(rs.mean()), float(rs.std())
    return (relevance(q_vec, p_vecs) - mu) / max(sd, eps), mu, sd


def lift_closed_form(q_vec, p_vecs, centroid, cov, eps: float = 1e-6):
    """Lift without looping over the pool, for unit vectors:
    mu_R = q . centroid,  sigma_R^2 = q^T Cov q.
    Step 1 uses this to check how close Lift is to plain relevance."""
    mu = float(q_vec @ centroid)
    sd = float(np.sqrt(max(q_vec @ cov @ q_vec, 0.0)))
    return (relevance(q_vec, p_vecs) - mu) / max(sd, eps), mu, sd


def compute_all(q_embs, corp_embs, ret_idx, pool_idx, eps: float = 1e-6) -> dict:
    """All three vector signals for every question. Returns dict of arrays."""
    pool_vecs = corp_embs[pool_idx]
    out = {k: [] for k in ["lift", "relevance", "consistency", "rand_mu", "rand_sd"]}
    for i in range(len(q_embs)):
        qv, pv = q_embs[i], corp_embs[ret_idx[i]]
        l, mu, sd = lift(qv, pv, pool_vecs, pool_idx, ret_idx[i], eps)
        out["lift"].append(l)
        out["relevance"].append(relevance(qv, pv))
        out["consistency"].append(consistency(pv))
        out["rand_mu"].append(mu)
        out["rand_sd"].append(sd)
    return {k: np.array(v) for k, v in out.items()}
