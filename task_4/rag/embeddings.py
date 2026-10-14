from __future__ import annotations

from sentence_transformers import SentenceTransformer


class NomicEmbedder:
    def __init__(
        self,
        model_name: str,
    ):
        self.model = SentenceTransformer(model_name)

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        vector = self.model.encode(
            f"search_query: {query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vector.tolist()
