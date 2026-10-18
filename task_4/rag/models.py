from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    title: str
    source: str
    chunk_index: int
    start_char: int
    similarity: float


@dataclass(frozen=True)
class RAGAnswer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    answered_from_kb: bool = True
