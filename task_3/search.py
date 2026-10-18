from __future__ import annotations

import sys

from .index_utils import create_chroma_client
from .index_utils import create_model
from .index_utils import get_collection

TOP_K = 5


def search(query: str, top_k: int = TOP_K) -> None:
    model = create_model()
    client = create_chroma_client()
    collection = get_collection(client)

    vector = model.encode(
        f"search_query: {query}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"QUERY: {query}")
    print("=" * 80)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        metadata = metadata or {}
        similarity = 1.0 - float(distance)
        print()
        print(f"#{index} similarity={similarity:.3f}")
        print(f"Title: {metadata.get('title', 'Unknown')}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Chunk: {metadata.get('chunk_index', -1)}")
        print("-" * 80)
        print((document or "")[:1000])


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m task_3.search "question"')
    search(" ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
