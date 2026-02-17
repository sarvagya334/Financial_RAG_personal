import os
import json
import pickle
from typing import Dict, List, Any

import faiss

from src.embeddings import build_faiss_index
from src.bm25_index import build_bm25


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save_jsonl(path: str, items: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def doc_country_keys(meta: dict) -> List[str]:
    """
    Which country buckets should this chunk belong to?
    - If meta["countries"]: bucket each + __MULTI__
    - If meta["country"]: bucket that
    - else: __UNKNOWN__
    """
    if meta.get("countries"):
        keys = [str(c) for c in meta["countries"] if c]
        keys.append("__MULTI__")
        return keys
    if meta.get("country"):
        return [str(meta["country"])]
    return ["__UNKNOWN__"]


def build_country_indexes(
    documents: List[Dict[str, Any]],
    embed_model,
    out_dir: str,
) -> List[str]:
    """
    Build FAISS+BM25 per country present in documents.
    Always rebuild (simple and safe).
    """
    faiss_dir = os.path.join(out_dir, "FAISS")
    bm25_dir = os.path.join(out_dir, "BM25")
    doc_dir = os.path.join(out_dir, "DOCMAP")

    _ensure_dir(faiss_dir)
    _ensure_dir(bm25_dir)
    _ensure_dir(doc_dir)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for d in documents:
        keys = doc_country_keys(d.get("metadata", {}))
        for k in keys:
            buckets.setdefault(k, []).append(d)

    built = []
    for country_key, docs in buckets.items():
        if not docs:
            continue

        texts = [d["text"] for d in docs]

        # FAISS
        index, _ = build_faiss_index(texts, embed_model)
        faiss_path = os.path.join(faiss_dir, f"{country_key}.index")
        faiss.write_index(index, faiss_path)

        # BM25
        bm25 = build_bm25(docs)
        bm25_path = os.path.join(bm25_dir, f"{country_key}.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

        # DOCMAP
        docmap_path = os.path.join(doc_dir, f"{country_key}.jsonl")
        _save_jsonl(docmap_path, docs)

        built.append(country_key)

    return built


def load_country_indexes(out_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads all per-country indexes present on disk.
    """
    faiss_dir = os.path.join(out_dir, "FAISS")
    bm25_dir = os.path.join(out_dir, "BM25")
    doc_dir = os.path.join(out_dir, "DOCMAP")

    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(faiss_dir):
        return out

    for file in os.listdir(faiss_dir):
        if not file.endswith(".index"):
            continue
        country_key = file.replace(".index", "")

        faiss_path = os.path.join(faiss_dir, file)
        bm25_path = os.path.join(bm25_dir, f"{country_key}.pkl")
        docmap_path = os.path.join(doc_dir, f"{country_key}.jsonl")

        if not (os.path.exists(bm25_path) and os.path.exists(docmap_path)):
            continue

        idx = faiss.read_index(faiss_path)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        docs = _load_jsonl(docmap_path)

        out[country_key] = {"faiss": idx, "bm25": bm25, "docs": docs}

    return out
