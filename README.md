# HALT-RAG

An evidence check that runs **before** a RAG system answers. Six cheap signals
(Lift, relevance, consistency, keyword coverage, clarity, CAS) go into a logistic
regression (the Signal Weight Learner), and a gate decides: **answer**, **search again**, or **refuse**.

Student: Shaithra Vilvarashah · Supervisor: Dr. Kasuni Welihinda

## Repo layout

```
config.yaml            all settings in one place (seed, sizes, paths)
requirements.txt
src/halt_rag/
  config.py            loads config.yaml
  data/
    popqa.py           answers, Wikidata IDs, head/tail sampling
    kilt_scan.py       one pass over KILT: match by Wikidata ID, random background
    passages.py        paragraphs -> ~100-word passages, remove duplicates
  retrieval/dense.py   MiniLM + exact FAISS search, Recall@k, MRR
  signals/
    embedding.py       Lift (with max(sigma, eps)), relevance, consistency
    lexical.py         keyword coverage, clarity
    cas.py             CAS (SQuAD 2.0 reader)
  eval/metrics.py      AUROC with bootstrap range, AUPRC, head/tail tables
notebooks/
  01_quick_checks.ipynb        Plan Step 1: is Lift more than relevance?
  02_kilt_scan_wikidata.ipynb  Plan Step 2: re-scan KILT matching by Wikidata ID
tests/test_core.py     quick checks of the maths (python -m pytest tests)
```

The pilot notebooks (`HALT_RAG_v5_*.ipynb`, `compute_lift.ipynb`,
`HALT_RAG_abstention_eval_1.ipynb`) are kept as they are, for the record.

## Lift

$$\mathrm{Lift}(q) = \frac{\overline{\mathrm{sim}}_{\text{top-}k}(q) - \mu_R(q)}{\max(\sigma_R(q), \epsilon)}$$

$R$ = one random sample of corpus passages, shared by all questions, with each
question's own retrieved passages removed.

## How to run (Colab)

Data lives in Google Drive, in `MyDrive/halt_rag_cache/` (not in git).

1. Push your latest commit, so Colab clones the same code.
2. Open a notebook from `notebooks/` in Colab (File → Open → GitHub, branch `lift_signal`).
3. Run all. The first cell clones the repo and mounts Drive.

| Notebook | Needs in Drive | Writes to Drive |
|---|---|---|
| 01_quick_checks | `halt_rag_v5_signals.csv` | `step1_quick_checks.json` |
| 02_kilt_scan_wikidata | `halt_rag_audit_v5.csv` | `kilt_cache_v6.pkl`, `kilt_scan_v6_report.json`, `halt_rag_audit_v6.csv` |

To feed the v6 scan into the signals step, point the v5 signals notebook's cache
path at `kilt_cache_v6.pkl` (same keys as v5).

## What changed from the pilot code

| Area | Pilot (v5) | Now |
|---|---|---|
| Page matching | title only | Wikidata ID first, title as backup |
| Lift bottom line | `sd if sd > 0 else 0` | `max(sd, eps)` |
| Passages | each paragraph cut on its own | short paragraphs joined up to ~100 words; duplicates removed |
| Scan report | found count only | also duplicates, records without Wikidata ID, match method |
