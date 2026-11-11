"""Evaluation helpers: AUROC with a bootstrap range, AUPRC, signal correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc_ci(y, score, n_boot: int = 2000, ci: float = 0.95, seed: int = 42):
    """AUROC plus a percentile bootstrap range. Returns (auc, low, high).
    Returns NaNs if y has only one class."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    ok = ~np.isnan(score) & ~pd.isna(y)
    y, score = y[ok].astype(int), score[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    boots, n = [], len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        boots.append(roc_auc_score(y[i], score[i]))
    a = (1 - ci) / 2
    return float(auc), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def auprc(y, score) -> float:
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    ok = ~np.isnan(score) & ~pd.isna(y)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return float(average_precision_score(y[ok].astype(int), score[ok]))


def signal_table(df: pd.DataFrame, signals, target: str, band_col: str = "band",
                 n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """One row per signal: AUROC vs the label, AUROC vs popularity (head=1),
    and a flag when the signal tracks popularity more than the label."""
    head = (df[band_col] == "head").astype(int)
    rows = []
    for s in signals:
        auc, lo, hi = auroc_ci(df[target], df[s], n_boot, seed=seed)
        b_auc, b_lo, b_hi = auroc_ci(head, df[s], n_boot, seed=seed)
        rows.append({
            "signal": s, "auroc_label": auc, "label_low": lo, "label_high": hi,
            "auprc_label": auprc(df[target], df[s]),
            "auroc_popularity": b_auc, "pop_low": b_lo, "pop_high": b_hi,
            # flagged only when the popularity AUROC is clearly away from 0.5 (its range excludes 0.5)
            "popularity_detector": bool(abs(b_auc - 0.5) > abs(auc - 0.5)
                                        and not (b_lo <= 0.5 <= b_hi)),
        })
    return pd.DataFrame(rows).set_index("signal")


def within_band_auroc(df, signals, target, band_col="band", n_boot=2000, seed=42):
    """AUROC inside head only and inside tail only (Objective 3, check 2)."""
    rows = []
    for band in ["head", "tail"]:
        sub = df[df[band_col] == band]
        for s in signals:
            auc, lo, hi = auroc_ci(sub[target], sub[s], n_boot, seed=seed)
            rows.append({"band": band, "signal": s, "n": len(sub),
                         "positives": int(sub[target].sum()), "auroc": auc,
                         "low": lo, "high": hi})
    return pd.DataFrame(rows).pivot(index="signal", columns="band",
                                    values=["auroc", "low", "high"])


def correlation_matrix(df, signals, method: str = "spearman") -> pd.DataFrame:
    return df[list(signals)].corr(method=method)
