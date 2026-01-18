import os
from typing import List, Dict
from .chunking import chunk_markdown_table_safe

def ingest_markdown(file_path: str, metadata: dict | None = None):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    auto_meta = infer_metadata_from_filename(os.path.basename(file_path))
    meta = {
        "source_file": os.path.basename(file_path),
        "source_type": "markdown",
        **auto_meta,
        **(metadata or {}),
    }

    docs = []
    chunks = chunk_markdown_table_safe(text, chunk_size=2500, overlap_blocks=1)


    for i, c in enumerate(chunks, start=1):
        docs.append({
            "text": c,
            "metadata": {
                **meta,
                "chunk_id": i,
                "chunk_total": len(chunks)
            }
        })

    return docs


def infer_metadata_from_filename(filename: str) -> dict:
    name = filename.lower()

    meta = {
        "country": None,
        "countries": None,
        "currency": None,
        "asset_class": "Stable",     # default
        "instrument": "Converted Doc",
        "regulator": None,
    }

    # cross-country
    if "comparison" in name or ("india" in name and "singapore" in name):
        meta["countries"] = ["India", "Singapore"]
        meta["asset_class"] = "Stable"
        return meta

    # single country
    if "india" in name:
        meta["country"] = "India"
        meta["currency"] = "INR"
        meta["regulator"] = "RBI"

    if "singapore" in name:
        meta["country"] = "Singapore"
        meta["currency"] = "SGD"
        meta["regulator"] = "MAS"

    return meta
