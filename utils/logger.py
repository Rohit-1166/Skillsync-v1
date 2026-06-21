import logging
import warnings
from pathlib import Path

from config.settings import LOG_DIR


# Suppress noisy Python warnings that external libraries emit during
# normal operation. Keeps console output clean during batch runs
# without masking genuine application-level errors.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Silence verbose INFO/DEBUG output from third-party libraries
# that would otherwise drown out SkillSync's own pipeline logs.
# ERROR level is retained so genuine failures from these libraries
# are still surfaced without manual log inspection.
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Log directory is created at import time so the file handler
# below never fails on a missing directory during first run.
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "skillsync.log"

# Named logger scopes all SkillSync output under a single namespace,
# making it easy to filter logs by source in multi-library environments.
logger = logging.getLogger("SkillSync")
logger.setLevel(logging.INFO)

# Prevent log records from bubbling up to the root logger,
# which would cause duplicate output if the root logger has handlers.
logger.propagate = False

# Clear any handlers that may have been registered by a previous
# import in the same Python process — common in hot-reload environments
# like FastAPI dev mode where modules are re-imported without restart.
if logger.hasHandlers():
    logger.handlers.clear()

# File handler provides a persistent verbose audit trail of all
# pipeline events, including timestamps for performance analysis
# and post-run debugging of ranking or parsing failures.
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler uses a minimal formatter to keep terminal output
# readable during development without repeating timestamp noise
# already captured in the log file.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


# Custom formatter strips log metadata from console output and
# replaces log levels with visual indicators so errors and warnings
# are immediately distinguishable during pipeline execution.
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