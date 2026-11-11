"""CAS: how strongly a SQuAD 2.0 reader points at a real answer instead of
its "no answer" slot (same as v5 cell 7).

per passage:  best_span = max start[a] + end[b] over valid spans in the passage
              null      = start[CLS] + end[CLS]
              score     = best_span - null        (higher = more answerable)
cas      = max over the top-k passages
cas_mean = mean over the top-k passages
"""
from __future__ import annotations

import numpy as np


class CASScorer:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2",
                 device: str | None = None, max_answer_len: int = 30):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        self.max_answer_len = max_answer_len

    def passage_scores(self, question: str, passages, max_len: int = 384, topn: int = 20):
        passages = [p for p in passages if str(p).strip()]
        if not passages:
            return np.array([-1e4])
        with self.torch.no_grad():
            enc = self.tok([question] * len(passages), list(passages), return_tensors="pt",
                           truncation=True, max_length=max_len, padding=True)
            out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        sl = out.start_logits.cpu().numpy()
        el = out.end_logits.cpu().numpy()

        scores = []
        for k in range(len(passages)):
            seq = enc.sequence_ids(k)
            ctx = np.array([j for j, s in enumerate(seq) if s == 1])
            null = float(sl[k, 0] + el[k, 0])
            if len(ctx) == 0:
                scores.append(-1e4)
                continue
            s_ctx, e_ctx = sl[k, ctx], el[k, ctx]
            best = -1e9
            for a in np.argsort(-s_ctx)[:topn]:
                for b in np.argsort(-e_ctx)[:topn]:
                    if b < a or (b - a + 1) > self.max_answer_len:
                        continue
                    best = max(best, float(s_ctx[a] + e_ctx[b]))
            if best <= -1e8:
                best = null - 1.0
            scores.append(best - null)
        return np.array(scores)

    def __call__(self, question: str, passages) -> tuple[float, float]:
        s = self.passage_scores(question, passages)
        return float(s.max()), float(s.mean())
