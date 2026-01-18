from langchain_openai.chat_models import ChatOpenAI
from .config import NVIDIA_BASE_URL, NVIDIA_MODEL, DEFAULT_TEMPERATURE
from .retrieval import retrieve_evidence
from .prompt import build_explainable_prompt
from .validators import validate_structure, validate_source_usage

def init_llm(nvidia_api_key: str):
    return ChatOpenAI(
        api_key=nvidia_api_key,
        base_url=NVIDIA_BASE_URL,
        model=NVIDIA_MODEL,
        temperature=DEFAULT_TEMPERATURE
    )

def rag_query(query, asset_class, model, index, documents, llm, countries=None):
    docs, err = retrieve_evidence(
        query=query,
        asset_class=asset_class,
        countries=countries,
        model=model,
        index=index,
        documents=documents
    )

    if err:
        return err

    messages = build_explainable_prompt(query, docs)
    answer = llm.invoke(messages).content

    if not validate_structure(answer):
        return "Answer rejected: missing required explainability sections."
    if not validate_source_usage(answer, docs):
        return "Answer rejected: references invalid sources."

    return answer
