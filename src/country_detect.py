import re
from typing import Optional, Dict

COUNTRY_ALIASES: Dict[str, str] = {
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "united states": "United States",
    "america": "United States",

    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",

    "uae": "United Arab Emirates",
}

def normalize_country(name: str) -> str:
    key = name.strip().lower()
    return COUNTRY_ALIASES.get(key, name.strip().title())

def detect_country_from_query(query: str) -> Optional[str]:
    q = query.lower()

    # direct aliases
    for alias, canon in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return canon

    # "in X"
    m = re.search(r"\bin\s+([a-zA-Z .]{3,40})\b", query)
    if m:
        candidate = m.group(1).strip(" .")
        if len(candidate) >= 3:
            return normalize_country(candidate)

    # "X market"
    m = re.search(r"\b([a-zA-Z .]{3,40})\s+market\b", query)
    if m:
        candidate = m.group(1).strip(" .")
        if len(candidate) >= 3:
            return normalize_country(candidate)

    return None
