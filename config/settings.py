from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

CANDIDATE_FILE = DATA_DIR / "candidates.jsonl"
JD_FILE = DATA_DIR / "Job_Description.pdf"

LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K_RETRIEVAL = 1000
TOP_K_FINAL = 100