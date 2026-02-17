from typing import List, Dict, Any, Optional
import numpy as np

from src.bm25_index import tokenize


def _minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.ones_like(x) * 0.5
    return (x - mn) / (mx - mn)


def retrieve_hybrid(
    query: str,
    *,
    documents: List[Dict[str, Any]],
    bm25,
    embed_model,
    faiss_index,
    top_k: int = 15,
    alpha: float = 0.65,
    faiss_k: int = 80,
    bm25_k: int = 120,
    per_file_limit: int = 4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
    - FAISS semantic retrieval
    - BM25 lexical retrieval
    - score fusion
    - per_file diversity limit
    """
    if not documents:
        return []

    # --- FAISS ---
    qvec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(qvec, faiss_k)

    vec_scores = {}
    for score, idx in zip(D[0], I[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(documents):
            continue
        vec_scores[idx] = float(score)

    # --- BM25 ---
    qtok = tokenize(query)
    bm_scores_all = bm25.get_scores(qtok)
    bm_sorted = np.argsort(bm_scores_all)[::-1][:bm25_k]
    bm_scores = {int(i): float(bm_scores_all[int(i)]) for i in bm_sorted if 0 <= int(i) < len(documents)}

    # --- Union candidates ---
    candidates = list(set(vec_scores.keys()) | set(bm_scores.keys()))
    if not candidates:
        return []

    vec_arr = np.array([vec_scores.get(i, 0.0) for i in candidates], dtype=np.float32)
    bm_arr = np.array([bm_scores.get(i, 0.0) for i in candidates], dtype=np.float32)

    vec_norm = _minmax(vec_arr)
    bm_norm = _minmax(bm_arr)

    hybrid = alpha * vec_norm + (1 - alpha) * bm_norm
    ranked = sorted(zip(candidates, hybrid), key=lambda x: x[1], reverse=True)

    # --- Diversity by file ---
    out = []
    file_count = {}

    for idx, score in ranked:
        src = documents[idx].get("metadata", {}).get("source_file", "unknown")
        file_count[src] = file_count.get(src, 0) + 1
        if file_count[src] <= per_file_limit:
            d = documents[idx]
            out.append({
                "text": d["text"],
                "metadata": d["metadata"],
                "hybrid_score": float(score),
                "vector_score": float(vec_scores.get(idx, 0.0)),
                "bm25_score": float(bm_scores.get(idx, 0.0)),
            })
        if len(out) >= top_k:
            break

    return out
