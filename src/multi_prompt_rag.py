from typing import List, Dict, Any
import json

# ---------- PROMPTS ----------

PROMPT_DECOMPOSE = """
You are given a user question. Break it into 6-10 smaller retrieval-friendly sub-questions.
Output strict JSON list with keys: id, sub_question, keywords.
User question: {question}
"""

PROMPT_EXTRACT = """
You are answering a sub-question using ONLY the provided context chunks.

Sub-question:
{sub_question}

Context chunks:
{context}

Rules:
- Use only context; do not guess.
- If no evidence, say "Not found in documents."
- Extract facts as bullet points.
Return strict JSON with keys:
answer_bullets (list of strings), confidence (0-100 int), missing_info (list of strings).
"""

PROMPT_AGGREGATE = """
You are given answers to multiple sub-questions.
Merge them into a final answer to the original question.

Original question:
{question}

Sub-answers JSON:
{sub_answers}

Rules:
- Remove duplicates
- If contradictions exist, mention them
- Output in clear sections with headings
"""

def format_context(chunks: List[Dict[str, Any]]) -> str:
    out = []
    for i, ch in enumerate(chunks, start=1):
        meta = ch.get("metadata", {})
        src = meta.get("source_file", "unknown")
        cid = meta.get("chunk_id", "?")
        out.append(f"[{i}] (src={src}, chunk={cid})\n{ch['text']}")
    return "\n\n".join(out)


def llm_json(llm, prompt: str) -> Any:
    """call LLM and parse json safely"""
    raw = llm(prompt)
    try:
        return json.loads(raw)
    except Exception:
        # fallback: try extracting json substring
        start = raw.find("[")
        if start == -1:
            start = raw.find("{")
        end = raw.rfind("]")
        if end == -1:
            end = raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
        raise ValueError("LLM returned non-JSON")


def multi_prompt_rag(
    question: str,
    retrieve_fn,        # function(q)->chunks
    llm,
    per_sub_top_k: int = 12
) -> str:

    # 1) Decompose question
    subqs = llm_json(llm, PROMPT_DECOMPOSE.format(question=question))

    sub_answers = []

    # 2) For each subquestion: retrieve + extract
    for item in subqs:
        sq = item["sub_question"]

        chunks = retrieve_fn(sq, top_k=per_sub_top_k)
        context = format_context(chunks)

        out = llm_json(llm, PROMPT_EXTRACT.format(
            sub_question=sq,
            context=context
        ))

        sub_answers.append({
            "id": item["id"],
            "sub_question": sq,
            "keywords": item.get("keywords", []),
            "retrieved": len(chunks),
            "result": out,
        })

    # 3) Aggregate
    final = llm(PROMPT_AGGREGATE.format(
        question=question,
        sub_answers=json.dumps(sub_answers, indent=2)
    ))

    return final
