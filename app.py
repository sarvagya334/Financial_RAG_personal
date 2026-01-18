import os

from src.config import DATA_RAW, DATA_PROCESSED
from src.convert_docling import convert_all_raw_to_markdown
from src.ingestion import ingest_markdown
from src.embeddings import load_embedding_model, build_faiss_index
from src.rag import init_llm, rag_query
from dotenv import load_dotenv

load_dotenv()  # loads .env into environment variables

nvidia_key = os.environ.get("NVIDIA_API_KEY")
if not nvidia_key:
    raise ValueError("NVIDIA_API_KEY not found in .env")


def main():
    # ---- API key ----
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise ValueError("Set NVIDIA_API_KEY in environment!")

    # ---- 1) Convert ALL raw documents to markdown ----
    md_files = convert_all_raw_to_markdown(DATA_RAW, DATA_PROCESSED)
    print("\nConverted markdown files:", len(md_files))

    # ---- 2) Ingest markdown docs only ----
    documents = []
    for md in md_files:
        documents += ingest_markdown(md)



    print("Total documents:", len(documents))

    # ---- 3) Embeddings + FAISS ----
    embed_model = load_embedding_model()
    texts = [d["text"] for d in documents]
    index, _ = build_faiss_index(texts, embed_model)

    # ---- 4) LLM ----
    llm = init_llm(nvidia_key)

    # ---- 5) Test query ----
    ans = rag_query(
        "what is inflation rate in India in 2024",
        asset_class="Stable",
        countries=["India", "Singapore"],
        model=embed_model,
        index=index,
        documents=documents,
        llm=llm
    )

    print("\n--- ANSWER ---\n")
    print(ans)


if __name__ == "__main__":
    main()
