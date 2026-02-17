from __future__ import annotations
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME
import numpy as np
import faiss
import os


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"     # ✅ Force CPU on mac
    except Exception:
        return "cpu"


def load_embedding_model() -> SentenceTransformer:
    device = detect_device()
    print(f"[Embeddings] Using device: {device}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def build_faiss_index(texts, model):
    # ✅ avoid loky/tokenizer multiprocessing issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    # ✅ cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings