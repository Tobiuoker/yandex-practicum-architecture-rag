from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .config import BATCH_SIZE
from .config import CHROMA_HOST
from .config import CHROMA_PORT
from .config import CHUNK_OVERLAP
from .config import CHUNK_SIZE
from .config import COLLECTION_NAME
from .config import MODEL_NAME
from .config import UPLOAD_BATCH_SIZE


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    embedding_text: str
    metadata: dict


def create_model() -> SentenceTransformer:
    """Load the local embedding model."""
    return SentenceTransformer(MODEL_NAME)


def create_splitter(
    model: SentenceTransformer,
) -> RecursiveCharacterTextSplitter:
    """Measure chunk size in the embedding model's tokens."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda text: len(
            model.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        ),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def get_title(text: str, fallback: str) -> str:
    """Return the first Markdown H1."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def get_markdown_files(directory: Path) -> list[Path]:
    """Return all Markdown documents in stable order."""
    files = sorted(directory.glob("*.md"))
    if not files:
        raise RuntimeError(f"No .md files found in {directory}")
    return files


def build_document_chunks(
    path: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> list[ChunkRecord]:
    """Split one document and prepare IDs, text and metadata."""
    text = path.read_text(encoding="utf-8")
    title = get_title(text, path.stem)
    documents = splitter.create_documents([text])

    chunks: list[ChunkRecord] = []
    for chunk_index, document in enumerate(documents):
        chunk = document.page_content.strip()
        if not chunk:
            continue

        chunks.append(
            ChunkRecord(
                id=f"{path.stem}_chunk_{chunk_index:04d}",
                text=chunk,
                embedding_text=f"search_document: {title}\n\n{chunk}",
                metadata={
                    "source": path.name,
                    "title": title,
                    "chunk_index": chunk_index,
                    "start_char": document.metadata.get("start_index", 0),
                },
            )
        )

    return chunks


def build_all_chunks(
    files: list[Path],
    splitter: RecursiveCharacterTextSplitter,
) -> list[ChunkRecord]:
    """Build chunks for all documents."""
    chunks: list[ChunkRecord] = []
    for path in files:
        chunks.extend(build_document_chunks(path, splitter))
    return chunks


def encode_chunks(
    model: SentenceTransformer,
    chunks: list[ChunkRecord],
    *,
    show_progress_bar: bool,
):
    """Generate normalized embeddings for prepared chunks."""
    if not chunks:
        raise RuntimeError("No chunks were created.")

    return model.encode(
        [chunk.embedding_text for chunk in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def create_chroma_client():
    """Connect to the external ChromaDB server."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    client.heartbeat()
    return client


def recreate_collection(client):
    """Delete the old collection and create a cosine collection."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def get_collection(client):
    """Return the existing collection."""
    return client.get_collection(COLLECTION_NAME)


def add_chunks(collection, chunks: list[ChunkRecord], embeddings) -> None:
    """Add a full set of chunks in batches."""
    for start in range(0, len(chunks), UPLOAD_BATCH_SIZE):
        end = start + UPLOAD_BATCH_SIZE
        batch = chunks[start:end]

        collection.add(
            ids=[item.id for item in batch],
            documents=[item.text for item in batch],
            metadatas=[item.metadata for item in batch],
            embeddings=embeddings[start:end].tolist(),
        )


def upsert_chunks(collection, chunks: list[ChunkRecord], embeddings) -> None:
    """Upsert chunks of one new or changed document."""
    if not chunks:
        return

    collection.upsert(
        ids=[item.id for item in chunks],
        documents=[item.text for item in chunks],
        metadatas=[item.metadata for item in chunks],
        embeddings=embeddings.tolist(),
    )


def file_hash(path: Path) -> str:
    """Calculate SHA-256 for change detection."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path) -> dict[str, str]:
    """Load indexed file hashes."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, str]) -> None:
    """Persist indexed file hashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
