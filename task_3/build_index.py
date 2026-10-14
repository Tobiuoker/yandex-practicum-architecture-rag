from __future__ import annotations

import json
import time

from .config import CHROMA_HOST
from .config import CHROMA_PORT
from .config import CHUNK_OVERLAP
from .config import CHUNK_SIZE
from .config import COLLECTION_NAME
from .config import INDEX_STATE_FILE
from .config import KNOWLEDGE_BASE
from .config import MODEL_NAME
from .config import STATS_FILE
from .index_utils import add_chunks
from .index_utils import build_all_chunks
from .index_utils import create_chroma_client
from .index_utils import create_model
from .index_utils import create_splitter
from .index_utils import encode_chunks
from .index_utils import file_hash
from .index_utils import get_markdown_files
from .index_utils import recreate_collection
from .index_utils import save_state


def save_statistics(
    *,
    documents: int,
    chunks: int,
    embedding_dimension: int,
    embedding_seconds: float,
    total_seconds: float,
) -> None:
    """Save reproducible build statistics."""
    stats = {
        "model": MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "documents": documents,
        "chunks": chunks,
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "embedding_seconds": round(embedding_seconds, 2),
        "total_seconds": round(total_seconds, 2),
        "collection": COLLECTION_NAME,
    }
    STATS_FILE.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    total_started = time.perf_counter()

    model = create_model()
    splitter = create_splitter(model)
    files = get_markdown_files(KNOWLEDGE_BASE)
    chunks = build_all_chunks(files, splitter)

    print(f"Documents: {len(files)}")
    print(f"Chunks: {len(chunks)}")
    print("Generating embeddings...")

    embedding_started = time.perf_counter()
    embeddings = encode_chunks(model, chunks, show_progress_bar=True)
    embedding_seconds = time.perf_counter() - embedding_started

    print(f"Embeddings generated in {embedding_seconds:.2f} sec")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Connecting to ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")

    client = create_chroma_client()
    collection = recreate_collection(client)
    add_chunks(collection, chunks, embeddings)

    save_state(
        INDEX_STATE_FILE,
        {path.name: file_hash(path) for path in files},
    )

    total_seconds = time.perf_counter() - total_started
    save_statistics(
        documents=len(files),
        chunks=len(chunks),
        embedding_dimension=int(embeddings.shape[1]),
        embedding_seconds=embedding_seconds,
        total_seconds=total_seconds,
    )

    print()
    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Documents: {len(files)}")
    print(f"Chunks: {collection.count()}")
    print(f"Embedding time: {embedding_seconds:.2f} sec")
    print(f"Total time: {total_seconds:.2f} sec")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
