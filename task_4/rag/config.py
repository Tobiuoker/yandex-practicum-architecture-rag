from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class RAGSettings:
    chroma_host: str
    chroma_port: int
    chroma_collection: str

    embedding_model: str

    llm_model: str
    ollama_base_url: str
    llm_timeout: float

    security_mode: str

    retrieval_top_k: int
    context_max_chunks: int
    min_similarity: float
    max_similarity_drop: float

    @classmethod
    def from_env(cls) -> "RAGSettings":
        load_dotenv()

        return cls(
            chroma_host=os.getenv(
                "CHROMA_HOST",
                "localhost",
            ),
            chroma_port=int(
                os.getenv(
                    "CHROMA_PORT",
                    "8000",
                )
            ),
            chroma_collection=os.getenv(
                "CHROMA_COLLECTION",
                "witcher_lore",
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "nomic-ai/nomic-embed-text-v1.5",
            ),
            llm_model=os.getenv(
                "LLM_MODEL",
                "qwen2.5:3b",
            ),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ),
            llm_timeout=float(
                os.getenv(
                    "LLM_TIMEOUT",
                    "180",
                )
            ),
            security_mode=os.getenv(
                "RAG_SECURITY_MODE",
                "secured",
            ),
            retrieval_top_k=int(
                os.getenv(
                    "RAG_RETRIEVAL_TOP_K",
                    "5",
                )
            ),
            context_max_chunks=int(
                os.getenv(
                    "RAG_CONTEXT_MAX_CHUNKS",
                    "2",
                )
            ),
            min_similarity=float(
                os.getenv(
                    "RAG_MIN_SIMILARITY",
                    "0.40",
                )
            ),
            max_similarity_drop=float(
                os.getenv(
                    "RAG_MAX_SIMILARITY_DROP",
                    "0.12",
                )
            ),
        )
