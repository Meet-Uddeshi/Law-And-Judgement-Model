from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data and Output Paths
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
JSON_DIR = DATA_DIR / "Json"
NORMALIZED_DIR = DATA_DIR / "Normalized Data"

# Vector Database Paths
FAISS_FILE = OUTPUT_DIR / "legal_index.faiss"
META_FILE = OUTPUT_DIR / "legal_index_metadata.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for p in [OUTPUT_DIR, JSON_DIR, NORMALIZED_DIR]:
    p.mkdir(parents=True, exist_ok=True)
