from langchain_core.messages import SystemMessage, HumanMessage

def build_explainable_prompt(query, docs):
    source_blocks = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        source_blocks.append(f"""
[Source {i}]
File: {meta.get("source_file")}
Country: {meta.get("country") or meta.get("countries")}
Asset Class: {meta.get("asset_class")}
Instrument: {meta.get("instrument")}

Content:
{d["text"]}
""")

    system_prompt = """
You are a FINANCIAL RESEARCH ANALYST AI.

STRICT RULES (NO EXCEPTIONS):
1. Use ONLY information present in the provided sources.
2. EVERY factual statement MUST cite a source number.
3. If information is missing, explicitly say:
   "This information is not available in the provided sources."
4. If inflation is mentioned, you MUST:
   - discuss inflation sensitivity OR
   - explicitly state that inflation data is missing.
5. If a calculation is required:
   - show the formula
   - show the inputs
   - or state why calculation cannot be done.
6. You MUST list assumptions explicitly.
7. Confidence MUST reflect evidence strength (High / Medium / Low).
8. You MUST NOT use external or general knowledge.
9. If the question is comparative, BOTH sides must be addressed.
""".strip()

    human_prompt = f"""
QUESTION:
{query}

AVAILABLE SOURCES:
{''.join(source_blocks)}

MANDATORY RESPONSE FORMAT:
1. Summary
2. Step-by-step reasoning (cite sources)
3. Calculations (if any)
4. Assumptions
5. Sources cited (explicit)
6. Confidence level
""".strip()

    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
