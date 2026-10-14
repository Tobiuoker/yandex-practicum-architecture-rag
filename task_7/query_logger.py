from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path


def serialize_sources(sources) -> list[dict]:
    return [
        {
            "source": item.source,
            "title": item.title,
            "chunk": item.chunk_index,
            "similarity": round(item.similarity, 4),
        }
        for item in sources
    ]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


class QueryLogger:
    def __init__(
        self,
        path: Path,
    ):
        self.path = path

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(
        self,
        *,
        question: str,
        answer: str,
        sources,
        answered: bool,
        latency_ms: int,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": question,
            "answer": answer,
            "answer_length": len(answer),
            "answered": answered,
            "chunks_found": len(sources),
            "top_similarity": (sources[0].similarity if sources else None),
            "sources": serialize_sources(sources),
            "latency_ms": latency_ms,
        }

        append_jsonl(self.path, record)
