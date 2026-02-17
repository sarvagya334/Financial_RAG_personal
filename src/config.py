import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_FAISS = os.path.join(BASE_DIR, "data", "faiss")

# Create folders if missing
for p in [DATA_RAW, DATA_PROCESSED, DATA_FAISS]:
    os.makedirs(p, exist_ok=True)

# ---- Metadata Rules ----
VALID_COUNTRIES = {"India", "Singapore"}
VALID_ASSET_CLASSES = {"Stable", "Growth"}

AMBIGUOUS_TERMS = ["sgs"]  # can expand list later

# ---- Embedding model ----
EMBEDDING_MODEL_NAME = "mukaj/fin-mpnet-base"
  # falls back to cpu automatically

# ---- LLM Config (NVIDIA endpoint) ----
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama3-405b-instruct"
DEFAULT_TEMPERATURE = 0
