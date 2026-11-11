"""Dense retrieval: MiniLM vectors + exact FAISS search (same as v5 cell 4)."""
from __future__ import annotations

import numpy as np


def encode(model, texts, batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
    """Unit-length float32 vectors, so inner product = cosine similarity."""
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                        normalize_embeddings=True).astype("float32")


def build_index(embs: np.ndarray):
    import faiss  # imported here so the rest of the package works without faiss
    index = faiss.IndexFlatIP(embs.shape[1])   # flat = exact search, no approximation
    index.add(embs)
    return index


def search(index, q_embs: np.ndarray, k: int):
    """Returns (scores, ids), each of shape (n_questions, k)."""
    return index.search(q_embs, k)


def recall_at_k(ret_ids: np.ndarray, gold_sets: list[set]) -> np.ndarray:
    """1 if any retrieved passage id is in that question's gold set, else 0."""
    return np.array([int(bool(set(map(int, r)) & g)) for r, g in zip(ret_ids, gold_sets)])


def mrr(ret_ids: np.ndarray, gold_sets: list[set]) -> np.ndarray:
    """1/rank of the first gold passage (0 if none in the list)."""
    out = []
    for r, g in zip(ret_ids, gold_sets):
        rr = 0.0
        for rank, j in enumerate(r, start=1):
            if int(j) in g:
                rr = 1.0 / rank
                break
        out.append(rr)
    return np.array(out)
