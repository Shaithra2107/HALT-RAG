# HALT-RAG — notes for Claude

## Who and what
- Student: Shaithra Vilvarashah (RGU 2410211 / IIT 20232694). Supervisor: Dr. Kasuni Welihinda.
- Final-year Individual Research Project. Methodology chapter was submitted on 10 Nov 2026.
- HALT-RAG is an evidence check that runs BEFORE a RAG system answers.
  Six label-free signals (lift, relevance, consistency, kw_coverage, clarity, CAS)
  -> logistic regression (the "Signal Weight Learner") -> gate: answer / search again / refuse.
- Data: PopQA questions (head = popular, tail = little known) against a fixed Wikipedia collection.

## How to talk to the student
- Always explain in simple English. Short sentences, plain words, small examples.
- Say what you plan to do BEFORE making changes, and wait for a yes on anything big.
- Do not change the methodology chapter (it lives in Overleaf, not here).

## Repo rules
- Branch: `lift_signal`. The student commits and pushes; do not push for them unless asked.
- Never commit anything in `data/` (it is in .gitignore). Data comes from Google Drive `halt_rag_cache/`.
- Settings live in `config.yaml`; do not hard-code paths, seeds or sizes.
- Keep the old pilot notebooks (`HALT_RAG_v5_*.ipynb`, `compute_lift.ipynb`,
  `HALT_RAG_abstention_eval_1.ipynb`) unchanged — they are the record of the pilot.
- Run tests with `python -m pytest tests`.

## Where things run
- Laptop: i5-1135G7, 8 GB RAM, no NVIDIA GPU, Windows, VS Code with `.venv`.
  Runs: quick checks, encoding, FAISS, signals, CAS (slow but OK), model, evaluation, gate.
- Colab: only the heavy steps — streaming all of KILT / a Wikipedia dump, building a
  Pyserini BM25 index, a large judge model. Outputs are copied into `data/`.
- Notebooks detect laptop vs Colab in their first cell (`halt_rag.config.data_dir`).

## Lift (the main contribution)
Lift(q) = (mean sim(q, top-k) − mu_R(q)) / max(sigma_R(q), eps)
- R = one random sample of corpus passages, shared by all questions.
- Remove the question's own retrieved passages from R.
- With unit vectors: mu_R = q · centroid, sigma_R² = qᵀΣq (closed form, used to check Lift ≠ relevance).

## Decisions so far
- Match PopQA to Wikipedia by Wikidata ID (`s_uri` <-> `wikidata_info.wikidata_id`), title only as backup.
  The v5 title match found only 133 of 299 subject pages; the cause is NOT the 2019 date gap.
- Supervisor advice: move to an updated Wikipedia (2023 dump). First get the true missing count on KILT.
- Full study: 1,000 questions, head/tail balanced. Top-k = 5. Seed 42.
- Labels: automatic judge with the Joren et al. sufficiency prompt is the main label (name the model);
  string match is a baseline only (wrong on ~57% of positives in a manual sample).
  The 50 hand labels were dropped. Human check (150–200, two annotators, kappa) is still undecided.
- Second search ("search again"): BM25 over the SAME collection, using the SAME question, top-5;
  recompute signals; second pass is answer-or-refuse only (no third search).
- Splits: grouped by subject entity; scaler fit on train only; cut-offs chosen on validation only.
- Federated training = future work, not this project.
- Step 1 result (300 pilot questions, provisional labels): Lift adds no AUROC over relevance.
  Spearman(lift, relevance) = 0.783, so the 0.9 alarm did not fire, but the out-of-fold gain from
  adding Lift to relevance was -0.001 (y_judge), -0.004 (y_cov), +0.003 (y_gen) — zero on all three.
  On the 134 `evaluable` rows only CAS survives (0.786); lift 0.488, relevance 0.537 are coin flips,
  and 5 of 7 signals are flagged as popularity detectors. Small n, 101/134 positives, wide ranges.
  Re-check after Step 2 fixes the collection — do not change Lift's background sample before then.
  The Step 1 decision rule is now OR (gain <= 0.01 **or** corr > 0.9), not AND.

## The 13-step plan (status)
0. Supervisor + paperwork — in progress
1. Quick checks on the 300 pilot questions — DONE (`notebooks/01_quick_checks.ipynb`, ran on laptop; see the Step 1 result above)
2. Fix the collection by Wikidata ID, then the 2023 dump; pick 1,000 questions; corpus manifest (`notebooks/02_kilt_scan_wikidata.ipynb`, Colab) — NEXT
3. Retrieval check: Recall@5, Recall@10, MRR, head vs tail
4. Signals + baselines (top-1 score, score gap, NQC, WIG); Lift with random vs hard-negative background
5. Labels (judge, string-match baseline)
6. Train/test the Signal Weight Learner: AUROC + AUPRC with bootstrap ranges, DeLong, ablation, calibration
7. Popularity check (lead finding): AUROC vs label and vs popularity, within head / within tail, Fisher test
8. Gate + second search: two error types, risk–coverage curve, always-answer / two-way / three-way / oracle, trigger and rescue rates
9. End-to-end with a generator (subset)
10. Collection-size test (10K / 25K / 50K background)
11. Shared model across datasets (Qasper, Doc2Dial) — extension
12. Write-up
