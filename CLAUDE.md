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
- Supervisor advice: move to an updated Wikipedia (2023 dump). Done — see the Step 2 result below.
- The collection now comes from the 2023 Wikipedia dump (`wikimedia/wikipedia`, config `20231101.en`)
  built by `notebooks/02b_wiki2023_build.ipynb`. `notebooks/02_kilt_scan_wikidata.ipynb` is NO LONGER
  USED: KILT cannot be loaded with the current `datasets` library, and streaming it took ~9 hours.
  Keep the file for the record, but do not run it.
- `notebooks/03_signals_w23.ipynb` is a copy of the pilot `HALT_RAG_v5_signals.ipynb` pointed at the
  2023 collection. It still carries the pilot's inline code: switch 03 to `src/halt_rag`
  (`merge_into_passages`, `dedupe`, `embedding.lift`) before the 1,000-question run.
  Its `y_cov` is recomputed from its own top-5 retrieved passages (gold answer string present,
  word boundaries); the old `m3_covered` column is stale because it came from the small pilot index.
  `m2_status_w23` is used only for the `evaluable` flag (`found` / `answer_not_on_page`).
- Full study: 1,500 questions (750 head / 750 tail, including the 300 pilot). Top-k = 5. Seed 42.
  Full study bands: tail s_pop < 100 (bottom ~10% of PopQA), head s_pop > 500.
  Pilot tail was drawn from s_pop < 50, a subset of the full-study tail band.
  The methodology chapter says ≤ 50; the final report must state this change and the reason
  (only 356 tail entities exist below 50, not enough for 750). Head threshold is unchanged.
  PopQA s_pop values changed since the pilot was sampled: 3 pilot tail questions now have
  s_pop > 100, and 1 pilot head question now has s_pop ≤ 500 in the current PopQA download.
  One pilot entity (George VI) also had two questions; the second was dropped.
  Total 5 pilot questions dropped and replaced with new questions from the same band (seed 42).
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
- Step 2 result (2023 dump via `02b_wiki2023_build.ipynb`): 298 of 299 subject pages found —
  279 by page ID, 4 by plain title match inside the dump, 15 filled from Wikipedia at the
  1 Nov 2023 revision. Answer on the subject page 261, answer not on page 38, page missing 1.
  Source `wikimedia/wikipedia 20231101.en`, 6,407,814 articles, 10,000 random background pages.
  So the old 133/299 was a title-matching problem, not a date-gap problem.
  This lifts `evaluable` from 134/300 to 299/300, which is why Step 1 has to be re-run.
- Step 3 result (300 pilot questions, `notebooks/04_retrieval_and_hard_lift.ipynb`, run 1 in `results/`):
  Recall@5 0.863 overall (head 0.907, tail 0.819); MRR@10 0.835; subject page ranks #1 for 82% of questions.
  `lift_hard` did not beat `lift` on y_judge (gain +0.001) and is more popularity-linked (pop AUROC 0.69).
  `score_gap` is the strongest cheap signal: AUROC ~0.71 on all labels, +0.13 over relevance alone,
  Spearman with relevance −0.09 (nearly independent), popularity AUROC 0.33 (reversed = head-favoured).
  Labels are still provisional (flan-t5-base placeholder + string match). Re-check after full run.
  `score_gap` added to the signal set for notebook 05 onwards.
  Notebooks 03 and 04 have been switched to `src/halt_rag` preprocessing (`clean_paragraphs`,
  `merge_into_passages`, `dedupe`) and all outputs renamed to `_1500` names (see config.yaml).
  Notebook 02b input switched to `audit_1500`. Run order: 02b → 03 → 04 on Colab.
  Next: run 02b, 03, 04 on Colab with the 1,500-question audit, then notebook 05
  (signal set including CAS + score_gap).

## The 13-step plan (status)
0. Supervisor + paperwork — in progress
1. Quick checks on the 300 pilot questions — DONE, but RE-CHECK AFTER 03 (`notebooks/01_quick_checks.ipynb`;
   the Step 1 numbers above are from the old pilot collection, so they are provisional)
2. Collection from the 2023 Wikipedia dump — DONE (`notebooks/02b_wiki2023_build.ipynb`, Colab;
   see the Step 2 result above). Notebook 02 (KILT) is retired, do not run it.
   Still to do in this step: pick the 1,000 questions and write the corpus manifest.
3. Retrieval check: Recall@5, Recall@10, MRR, head vs tail — DONE (`notebooks/04_retrieval_and_hard_lift.ipynb`;
   see Step 3 result above). Also tested `lift_hard`, `gap_hard`, `score_gap`.
4. Signals + baselines (top-1 score, score gap, NQC, WIG); Lift with random vs hard-negative background
   — NEXT. First: switch 03 + 04 to `src/halt_rag` and scale to 1,500 questions; then notebook 05
   (signal set: CAS + score_gap + lift, on the full 1,500-question run).
5. Labels (judge, string-match baseline)
6. Train/test the Signal Weight Learner: AUROC + AUPRC with bootstrap ranges, DeLong, ablation, calibration
7. Popularity check (lead finding): AUROC vs label and vs popularity, within head / within tail, Fisher test
8. Gate + second search: two error types, risk–coverage curve, always-answer / two-way / three-way / oracle, trigger and rescue rates
9. End-to-end with a generator (subset)
10. Collection-size test (10K / 25K / 50K background)
11. Shared model across datasets (Qasper, Doc2Dial) — extension
12. Write-up
