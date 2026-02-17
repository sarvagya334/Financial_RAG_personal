import os
from dotenv import load_dotenv

from src.config import DATA_RAW, DATA_PROCESSED
from src.convert_docling import convert_all_raw_to_markdown
from src.ingestion import ingest_markdown
from src.embeddings import load_embedding_model
from src.country_indexes import build_country_indexes, load_country_indexes
from src.rag import init_llm, rag_query


# performance env vars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

load_dotenv()


def main():
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise ValueError("NVIDIA_API_KEY not found in .env")

    # ---- 1) Convert raw -> markdown ----
    md_files = convert_all_raw_to_markdown(DATA_RAW, DATA_PROCESSED)
    print("\nConverted markdown files:", len(md_files))

    # ---- 2) Ingest markdown (chunk once cache) ----
    documents = []
    chunk_cache_dir = os.path.join(DATA_PROCESSED, "chunks_cache")
    for md in md_files:
        documents.extend(ingest_markdown(md, cache_dir=chunk_cache_dir))

    print("Total chunks/documents:", len(documents))

    # ---- 3) Embedding model ----
    embed_model = load_embedding_model()

    # ---- 4) Build country indexes (FAISS + BM25 per country) ----
    country_index_dir = os.path.join(DATA_PROCESSED, "country_indexes")
    built = build_country_indexes(
        documents=documents,
        embed_model=embed_model,
        out_dir=country_index_dir,
    )
    if built:
        print("Built/updated country indexes:", built)

    country_indexes = load_country_indexes(country_index_dir)
    print("Loaded country indexes:", list(country_indexes.keys()))

    # ---- 5) LLM ----
    llm = init_llm(nvidia_key)

    # ---- 6) Query ----
    query = "what are curretn interest rate of national saving schemes from banks Inida"  # change this

    ans = rag_query(
        query=query,
        country_indexes=country_indexes,
        embed_model=embed_model,
        llm=llm,
    )

    print("\n--- ANSWER ---\n")
    print(ans)


if __name__ == "__main__":
    main()
