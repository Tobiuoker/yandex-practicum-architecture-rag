from __future__ import annotations

import chromadb

from .embeddings import NomicEmbedder
from .models import RetrievedChunk


class ChromaRetriever:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        collection_name: str,
        embedder: NomicEmbedder,
    ):
        self.embedder = embedder

        self.client = chromadb.HttpClient(
            host=host,
            port=port,
        )

        self.client.heartbeat()

        self.collection = self.client.get_collection(collection_name)

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks = []

        for (chunk_id, text, metadata, distance,) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            metadata = metadata or {}

            chunks.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    text=text or "",
                    title=str(
                        metadata.get(
                            "title",
                            "Unknown",
                        )
                    ),
                    source=str(
                        metadata.get(
                            "source",
                            "Unknown",
                        )
                    ),
                    chunk_index=int(
                        metadata.get(
                            "chunk_index",
                            -1,
                        )
                    ),
                    start_char=int(
                        metadata.get(
                            "start_char",
                            -1,
                        )
                    ),
                    similarity=(1.0 - float(distance)),
                )
            )

        return chunks


def select_context_chunks(
    chunks: list[RetrievedChunk],
    *,
    min_similarity: float,
    max_chunks: int,
    max_similarity_drop: float,
) -> list[RetrievedChunk]:
    """
    Keep only strong context.

    Small language models are more reliable when they receive a small number
    of highly relevant chunks instead of several loosely related fragments.
    """

    if not chunks:
        return []

    best_score = chunks[0].similarity

    if best_score < min_similarity:
        return []

    selected = []

    for chunk in chunks:
        if chunk.similarity < min_similarity:
            continue

        if best_score - chunk.similarity > max_similarity_drop:
            continue

        selected.append(chunk)

        if len(selected) >= max_chunks:
            break

    return selected
