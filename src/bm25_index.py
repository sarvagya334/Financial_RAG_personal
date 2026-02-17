import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokenize(text: str) -> List[str]:
    """
    Finance + table-friendly tokenizer.
    """
    text = normalize_text(text)
    return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?|\d+(?:\.\d+)?%?|\w+(?:-\w+)+", text)


def build_bm25(documents: List[Dict[str, Any]]):
    tokenized = [tokenize(d["text"]) for d in documents]
    return BM25Okapi(tokenized)
