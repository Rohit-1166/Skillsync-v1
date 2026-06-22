from pathlib import Path

# Resolve project root directory regardless of where the application is launched.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

CANDIDATE_FILE = DATA_DIR / "candidates.jsonl"
JD_FILE = DATA_DIR / "Job_Description.pdf"

# Create required directories automatically if they do not exist.
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Embedding model used for semantic candidate-job matching.
EMBEDDING_MODEL = str(BASE_DIR / "model" / "bge-small-en-v1.5")

# Number of candidates retrieved from FAISS before reranking.
TOP_K_RETRIEVAL = 1000

# Number of final candidates returned after ranking.
TOP_K_FINAL = 100