from __future__ import annotations
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME


def detect_device() -> str:
    """
    General device selection (works on any PC):
    Priority:
      1) CUDA GPU (NVIDIA)
      2) Apple Silicon MPS
      3) CPU fallback
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        # Apple Silicon support
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    except Exception:
        return "cpu"


def load_embedding_model() -> SentenceTransformer:
    device = detect_device()
    print(f"[Embeddings] Using device: {device}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def build_faiss_index(texts, model):
    import faiss

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings
