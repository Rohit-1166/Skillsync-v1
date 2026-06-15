import logging
import warnings
from pathlib import Path

from config.settings import LOG_DIR

# 1. Suppress Python warnings in console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Silence noise from external libraries
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "skillsync.log"

# 3. Create a dedicated project logger
logger = logging.getLogger("SkillSync")
logger.setLevel(logging.INFO)
logger.propagate = False  # Avoid root propagation duplicated printing

# Clear existing handlers to prevent duplicate binding
if logger.hasHandlers():
    logger.handlers.clear()

# 4. File Handler (Verbose Auditing logs)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 5. Console Handler (Beautiful, clean UI)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

class CleanConsoleFormatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"❌ Error: {msg}"
        elif record.levelno >= logging.WARNING:
            return f"⚠️ Warning: {msg}"
        else:
            return f"{msg}"

console_handler.setFormatter(CleanConsoleFormatter())
logger.addHandler(console_handler)
