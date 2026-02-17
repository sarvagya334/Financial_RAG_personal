import os
import re
from typing import List, Dict, Any

from .chunking import chunk_markdown_page_level

from .chunk_cache import file_sha256, cache_path_for, write_jsonl, read_jsonl


# -------------------------
# Topics
# -------------------------

TOPIC_RULES = {
    "interest_rate": [r"interest rate", r"\bp\.a\.\b", r"\brate\b", r"yield", r"coupon"],
    "tenure": [r"tenure", r"maturity", r"lock[- ]?in", r"years?", r"months?"],
    "tax": [r"tax", r"80c", r"tds", r"exempt", r"deduction"],
    "eligibility": [r"eligible", r"eligibility", r"resident", r"citizen", r"individual"],
    "liquidity": [r"withdraw", r"premature", r"exit", r"redemption", r"liquid"],
    "risk": [r"risk", r"guarantee", r"government backed", r"credit"],
    "gold": [r"gold", r"sovereign gold bond", r"\bsgb\b"],
    "inflation": [r"inflation", r"cpi"],
}


def infer_topics(text: str) -> List[str]:
    t = text.lower()
    tags = []
    for tag, patterns in TOPIC_RULES.items():
        for p in patterns:
            if re.search(p, t):
                tags.append(tag)
                break
    return sorted(set(tags))


# -------------------------
# Country metadata inference (scales)
# -------------------------

COUNTRY_KEYWORDS = {
    # Today:
    "india": {"country": "India", "currency": "INR", "regulator": "RBI"},
    "singapore": {"country": "Singapore", "currency": "SGD", "regulator": "MAS"},

    # Tomorrow scale (examples)
    "usa": {"country": "United States", "currency": "USD", "regulator": "SEC"},
    "united_states": {"country": "United States", "currency": "USD", "regulator": "SEC"},
    "united states": {"country": "United States", "currency": "USD", "regulator": "SEC"},
    "uk": {"country": "United Kingdom", "currency": "GBP", "regulator": "FCA"},
    "uae": {"country": "United Arab Emirates", "currency": "AED", "regulator": "SCA"},
}


def infer_metadata_from_filename(filename: str) -> dict:
    name = filename.lower()

    meta = {
        "country": None,
        "countries": None,
        "currency": None,
        "asset_class": "Stable",
        "instrument": "Converted Doc",
        "regulator": None,
    }

    # multi-country
    if "comparison" in name or ("india" in name and "singapore" in name):
        meta["countries"] = ["India", "Singapore"]
        meta["asset_class"] = "Stable"
        return meta

    # single country by keyword dictionary
    for key, vals in COUNTRY_KEYWORDS.items():
        if key in name:
            meta.update(vals)
            break

    return meta


def build_chunk_header(meta: dict) -> str:
    country = meta.get("country") or meta.get("countries") or "Unknown"
    regulator = meta.get("regulator") or "Unknown"
    currency = meta.get("currency") or "Unknown"
    src = meta.get("source_file") or "Unknown"
    return f"[COUNTRY={country} | CURRENCY={currency} | REGULATOR={regulator} | SOURCE={src}]"



def ingest_markdown(
    file_path: str,
    metadata: dict | None = None,
    *,
    cache_dir: str = None
) -> List[Dict[str, Any]]:
    """
    Chunking happens ONLY ONCE per markdown file (cached).
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(file_path), "_chunks_cache")
    os.makedirs(cache_dir, exist_ok=True)

    sha = file_sha256(file_path)
    cache_path = cache_path_for(file_path, cache_dir)

    # ---- load cache if unchanged ----
    if os.path.exists(cache_path):
        cached = read_jsonl(cache_path)
        if cached and cached[0].get("metadata", {}).get("file_sha256") == sha:
            return cached

    # ---- chunk once and cache ----
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    auto_meta = infer_metadata_from_filename(os.path.basename(file_path))
    meta = {
        "source_file": os.path.basename(file_path),
        "source_type": "markdown",
        "file_sha256": sha,
        **auto_meta,
        **(metadata or {}),
    }

    chunks = chunk_markdown_page_level(md_text)

    header = build_chunk_header(meta)

    docs: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        topics = infer_topics(c)

        enriched = header
        if topics:
            enriched += f"\n[TOPICS: {', '.join(topics)}]"
        enriched += "\n\n" + c

        docs.append({
            "text": enriched,
            "metadata": {
                **meta,
                "chunk_id": i,
                "chunk_total": len(chunks),
                "topics": topics,
            }
        })

    write_jsonl(cache_path, docs)
    return docs
