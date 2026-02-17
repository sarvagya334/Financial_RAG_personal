from typing import List, Dict, Any, Optional, Callable
import os
import requests

from src.hybrid_retrieval import retrieve_hybrid
from src.country_detect import detect_country_from_query


# -----------------------------
# LLM INIT (NVIDIA)
# -----------------------------

def init_llm(api_key: str, model: str = "meta/llama3-8b-instruct") -> Callable[[str], str]:
    """
    Returns callable llm(prompt)->string
    Override NVIDIA_BASE_URL in .env if needed.
    """
    base_url = os.environ.get("NVIDIA_BASE_URL", "").strip()
    if not base_url:
        base_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def llm(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful assistant that only answers from provided context."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 900,
        }
        r = requests.post(base_url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    return llm


# -----------------------------
# Helpers
# -----------------------------

def _format_context(chunks: List[Dict[str, Any]], max_chars: int = 14000) -> str:
    parts = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        meta = ch.get("metadata", {})
        src = meta.get("source_file", "unknown")
        cid = meta.get("chunk_id", "?")
        block = f"[{i}] source={src}, chunk={cid}\n{ch['text']}".strip()
        total += len(block)
        if total > max_chars:
            break
        parts.append(block)
    return "\n\n".join(parts)


def _group_by_source(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        src = ch.get("metadata", {}).get("source_file", "unknown")
        grouped.setdefault(src, []).append(ch)
    return grouped


# -----------------------------
# Main RAG Query (country-aware)
# -----------------------------

def rag_query(
    *,
    query: str,
    country_indexes: Dict[str, Dict[str, Any]],
    embed_model,
    llm,
) -> str:
    """
    Country-aware RAG:
    - If query requests a country and no index exists -> "No documents available"
    - Otherwise retrieve hybrid from that country
    - If no country specified, searches across all countries
    """

    requested = detect_country_from_query(query)

    def retrieve_from_pack(pack, q: str, top_k=18):
        return retrieve_hybrid(
            q,
            documents=pack["docs"],
            bm25=pack["bm25"],
            embed_model=embed_model,
            faiss_index=pack["faiss"],
            top_k=top_k,
            alpha=0.65,
            faiss_k=100,
            bm25_k=160,
            per_file_limit=4,
        )

    # --- if requested country present ---
    if requested:
        if requested not in country_indexes:
            return f"No documents available for country/market: {requested}"

        pack = country_indexes[requested]
        chunks = retrieve_from_pack(pack, query, top_k=18)

        if not chunks:
            return f"No relevant documents found for: {requested}"

        ctx = _format_context(chunks)

        prompt = f"""
You are a strict RAG assistant.
Answer ONLY using the context.
If answer isn't present, say "Not found in documents."

Question:
{query}

Context:
{ctx}

Answer in structured format and include key numbers if present.
"""
        return llm(prompt).strip()

    # --- no country requested: search ALL indexes and merge ---
    merged = []
    for _, pack in country_indexes.items():
        merged.extend(retrieve_from_pack(pack, query, top_k=8))

    if not merged:
        return "No relevant documents found."

    merged.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
    chunks = merged[:20]

    # Special: summarize all
    q_lower = query.lower()
    summarize_all = any(x in q_lower for x in [
        "summarize all documents",
        "summarise all documents",
        "summarize everything",
        "summarise everything",
        "summary of all documents",
    ])

    if summarize_all:
        grouped = _group_by_source(chunks)
        per_doc_summaries = []
        for src, src_chunks in grouped.items():
            ctx = _format_context(src_chunks, max_chars=9000)
            prompt = f"""
Summarize ONE document using ONLY context.

Document: {src}
Context:
{ctx}

Write 6-12 bullet points including:
- key numbers (rates, limits, tenures)
- eligibility/tax/liquidity if present
If missing, write "Not found".
"""
            per_doc_summaries.append(f"### {src}\n{llm(prompt).strip()}")

        merge_prompt = f"""
Combine the following document summaries into a consolidated report.

User question: {query}

Summaries:
{chr(10).join(per_doc_summaries)}

Output:
1) Consolidated summary (10-15 bullets)
2) Country-wise breakdown
3) Important numbers
4) Missing info
"""
        return llm(merge_prompt).strip()

    ctx = _format_context(chunks)
    prompt = f"""
You are a strict RAG assistant.
Answer ONLY using context.

Question:
{query}

Context:
{ctx}

Answer clearly and include numbers if present.
"""
    return llm(prompt).strip()
