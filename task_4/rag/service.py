from __future__ import annotations

import time
from pathlib import Path

from task_5.security import contains_sensitive_output
from task_5.security import filter_chunks
from task_7 import QueryLogger

from .config import RAGSettings
from .embeddings import NomicEmbedder
from .llm import OllamaLLM
from .models import RAGAnswer
from .prompts import UNKNOWN_ANSWER
from .prompts import build_instructions
from .prompts import build_user_prompt
from .retriever import ChromaRetriever
from .retriever import select_context_chunks


class RAGService:
    """
    Application service independent of Telegram, CLI or any other interface.
    """

    def __init__(
        self,
        settings: RAGSettings,
    ):
        self.settings = settings

        self.embedder = NomicEmbedder(settings.embedding_model)

        self.retriever = ChromaRetriever(
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection_name=settings.chroma_collection,
            embedder=self.embedder,
        )

        self.llm = OllamaLLM(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            timeout=settings.llm_timeout,
        )

        self.instructions = build_instructions(security_mode=settings.security_mode)

        self.query_logger = QueryLogger(Path("logs/queries.jsonl"))

    @classmethod
    def from_env(cls) -> "RAGService":
        return cls(RAGSettings.from_env())

    def ask(
        self,
        question: str,
    ) -> RAGAnswer:
        started = time.perf_counter()

        result = self._ask(question)

        latency_ms = int((time.perf_counter() - started) * 1000)

        self.query_logger.log(
            question=question,
            answer=result.text,
            sources=result.sources,
            answered=result.answered_from_kb,
            latency_ms=latency_ms,
        )

        return result

    def _ask(
        self,
        question: str,
    ) -> RAGAnswer:
        question = question.strip()

        if not question:
            return RAGAnswer(
                text="Введите непустой вопрос.",
                sources=[],
                answered_from_kb=False,
            )

        candidates = self.retriever.retrieve(
            question,
            top_k=self.settings.retrieval_top_k,
        )

        context_chunks = select_context_chunks(
            candidates,
            min_similarity=self.settings.min_similarity,
            max_chunks=self.settings.context_max_chunks,
            max_similarity_drop=self.settings.max_similarity_drop,
        )

        if self.settings.security_mode == "secured":
            context_chunks, _ = filter_chunks(context_chunks)

        if not context_chunks:
            return RAGAnswer(
                text=UNKNOWN_ANSWER,
                sources=[],
                answered_from_kb=False,
            )

        prompt = build_user_prompt(
            question,
            context_chunks,
        )

        answer = self.llm.generate(
            instructions=self.instructions,
            user_prompt=prompt,
        )

        if self.settings.security_mode == "secured" and contains_sensitive_output(answer):
            return RAGAnswer(
                text="Answer is blocked for security reasons",
                sources=context_chunks,
                answered_from_kb=False,
            )

        answered = answer.strip() != UNKNOWN_ANSWER

        return RAGAnswer(
            text=answer,
            # Show only the context actually supplied to the LLM.
            sources=context_chunks,
            answered_from_kb=answered,
        )
