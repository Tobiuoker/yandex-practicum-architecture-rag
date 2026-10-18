from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

KNOWLEDGE_BASE = PROJECT_ROOT / "task_2" / "knowledge_base"
INDEX_STATE_FILE = PROJECT_ROOT / "task_6" / ".index_state.json"
STATS_FILE = TASK_DIR / "index_stats.json"

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-ai/nomic-embed-text-v1.5",
)
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "witcher_lore")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
BATCH_SIZE = 32
UPLOAD_BATCH_SIZE = 128
