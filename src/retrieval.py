from typing import List, Dict, Optional
from .config import AMBIGUOUS_TERMS

def detect_ambiguity(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in AMBIGUOUS_TERMS)

def governed_search(query, country=None, asset_class=None, k=6, model=None, index=None, documents=None):
    q_emb = model.encode([query])
    _, idxs = index.search(q_emb, k * 6)

    results = []
    for idx in idxs[0]:
        doc = documents[idx]
        meta = doc["metadata"]

        # Country rule
        if country:
            if meta.get("country"):
                if meta["country"] != country:
                    continue
            elif meta.get("countries"):
                if country not in meta["countries"]:
                    continue
            else:
                continue

        # Asset class rule
        if asset_class and meta.get("asset_class") != asset_class:
            continue

        results.append(doc)
        if len(results) == k:
            break

    return results

def balanced_governed_search(query, countries, asset_class, k_per_country=3, model=None, index=None, documents=None):
    results = []
    for c in countries:
        results.extend(governed_search(
            query, country=c, asset_class=asset_class, k=k_per_country,
            model=model, index=index, documents=documents
        ))

    # cross country
    results.extend(governed_search(
        query, country=None, asset_class=asset_class, k=2,
        model=model, index=index, documents=documents
    ))
    return results

def validate_country_coverage(docs, countries):
    present = set()
    for d in docs:
        meta = d["metadata"]
        if meta.get("country"):
            present.add(meta["country"])
        if meta.get("countries"):
            present.update(meta["countries"])
    return all(c in present for c in countries)

def retrieve_evidence(query, asset_class, countries=None, model=None, index=None, documents=None):
    if detect_ambiguity(query):
        return None, (
            "Ambiguous instrument detected.\n"
            "• India: Sovereign Gold Bonds (SGB)\n"
            "• Singapore: Singapore Government Securities (SGS)\n"
            "Please clarify which one you mean."
        )

    if countries:
        docs = balanced_governed_search(
            query, countries=countries, asset_class=asset_class,
            model=model, index=index, documents=documents
        )
        if not validate_country_coverage(docs, countries):
            return None, "Insufficient balanced evidence for comparison."
    else:
        docs = governed_search(
            query, asset_class=asset_class, k=6,
            model=model, index=index, documents=documents
        )

    if not docs:
        return None, "No relevant data found in knowledge base."

    return docs, None
