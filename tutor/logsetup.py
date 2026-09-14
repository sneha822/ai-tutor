"""Console + file logging for the app scripts. Log files land in ./logs/ (overwritten each run)."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

_NOISY = ("httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb", "huggingface_hub",
          "faster_whisper", "phonemizer", "numba", "spacy", "uvicorn")


def setup(log_file: str | None = None, level: int = logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(name)-6s %(levelname)-5s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_file:
        LOG_DIR.mkdir(exist_ok=True)
        fh = logging.FileHandler(LOG_DIR / log_file, mode="w")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)  # warns once per phoneme on any non-English text
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch.*")
