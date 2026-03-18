from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default)).strip()
    try:
        return float(value)
    except ValueError:
        return default


_load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "raw"))
CHUNK_STORE_DIR = Path(os.getenv("CHUNK_STORE_DIR", "chunk_store"))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "vector_store"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "llama3.1:8b")

RETRIEVAL_TOP_K = _get_int("RETRIEVAL_TOP_K", 5)
MIN_RETRIEVAL_SCORE = _get_float("MIN_RETRIEVAL_SCORE", 0.55)
MULTI_QUERY_COUNT = _get_int("MULTI_QUERY_COUNT", 3)
