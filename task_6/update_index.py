from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from task_3.config import INDEX_STATE_FILE
from task_3.config import KNOWLEDGE_BASE
from task_3.index_utils import build_document_chunks
from task_3.index_utils import create_chroma_client
from task_3.index_utils import create_model
from task_3.index_utils import create_splitter
from task_3.index_utils import encode_chunks
from task_3.index_utils import file_hash
from task_3.index_utils import get_collection
from task_3.index_utils import load_state
from task_3.index_utils import save_state
from task_3.index_utils import upsert_chunks
from task_6.logging_utils import setup_logging

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"


logger = logging.getLogger(__name__)


def get_current_files() -> dict[str, Path]:
    return {path.name: path for path in KNOWLEDGE_BASE.glob("*.md")}


def detect_changes(
    old_state: dict[str, str],
    current_files: dict[str, Path],
) -> tuple[list[Path], list[Path], list[str]]:
    added: list[Path] = []
    changed: list[Path] = []

    for filename, path in current_files.items():
        current_hash = file_hash(path)
        old_hash = old_state.get(filename)

        if old_hash is None:
            added.append(path)
        elif old_hash != current_hash:
            changed.append(path)

    deleted = [filename for filename in old_state if filename not in current_files]

    return sorted(added), sorted(changed), sorted(deleted)


def delete_document(collection, filename: str) -> None:
    collection.delete(where={"source": filename})


def index_document(path: Path, model, splitter, collection) -> int:
    """Rebuild all chunks of one new or changed document."""
    chunks = build_document_chunks(path, splitter)
    embeddings = (
        encode_chunks(
            model,
            chunks,
            show_progress_bar=False,
        )
        if chunks
        else None
    )

    delete_document(collection, path.name)
    if chunks and embeddings is not None:
        upsert_chunks(collection, chunks, embeddings)

    return len(chunks)


def process_deleted_files(
    filenames: list[str],
    collection,
    state: dict[str, str],
) -> tuple[int, int]:
    processed = 0
    errors = 0

    for filename in filenames:
        try:
            delete_document(collection, filename)
            state.pop(filename, None)
            processed += 1
            logger.info("Deleted: %s", filename)
        except Exception:
            errors += 1
            logger.exception("Failed to delete: %s", filename)

    return processed, errors


def process_documents(
    files: list[Path],
    *,
    label: str,
    model,
    splitter,
    collection,
    state: dict[str, str],
) -> tuple[int, int, int]:
    processed = 0
    chunks_count = 0
    errors = 0

    for path in files:
        try:
            chunks = index_document(path, model, splitter, collection)
            state[path.name] = file_hash(path)
            processed += 1
            chunks_count += chunks
            logger.info("%s: %s | chunks=%d", label, path.name, chunks)
        except Exception:
            errors += 1
            logger.exception("Failed to process: %s", path.name)

    return processed, chunks_count, errors


def main() -> None:
    setup_logging(LOG_DIR / "update_index.log")
    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()

    logger.info(
        "Index update started | time=%s",
        started_at.isoformat(timespec="seconds"),
    )

    try:
        old_state = load_state(INDEX_STATE_FILE)
        new_state = dict(old_state)
        current_files = get_current_files()
        added, changed, deleted = detect_changes(old_state, current_files)

        logger.info(
            "Detected changes | added=%d | changed=%d | deleted=%d",
            len(added),
            len(changed),
            len(deleted),
        )

        client = create_chroma_client()
        collection = get_collection(client)

        deleted_files, delete_errors = process_deleted_files(
            deleted,
            collection,
            new_state,
        )

        model = create_model() if added or changed else None
        splitter = create_splitter(model) if model is not None else None

        added_files = new_chunks = added_errors = 0
        if added:
            added_files, new_chunks, added_errors = process_documents(
                added,
                label="Added",
                model=model,
                splitter=splitter,
                collection=collection,
                state=new_state,
            )

        changed_files = reindexed_chunks = changed_errors = 0
        if changed:
            changed_files, reindexed_chunks, changed_errors = process_documents(
                changed,
                label="Updated",
                model=model,
                splitter=splitter,
                collection=collection,
                state=new_state,
            )

        save_state(INDEX_STATE_FILE, new_state)

        errors = delete_errors + added_errors + changed_errors
        finished_at = datetime.now().astimezone()
        duration = time.perf_counter() - started_perf

        logger.info(
            "Index update completed | started=%s | finished=%s | "
            "added_files=%d | changed_files=%d | deleted_files=%d | "
            "new_chunks=%d | reindexed_chunks=%d | index_size=%d | "
            "errors=%d | duration=%.2fs",
            started_at.isoformat(timespec="seconds"),
            finished_at.isoformat(timespec="seconds"),
            added_files,
            changed_files,
            deleted_files,
            new_chunks,
            reindexed_chunks,
            collection.count(),
            errors,
            duration,
        )
    except Exception:
        finished_at = datetime.now().astimezone()
        duration = time.perf_counter() - started_perf
        logger.exception(
            "Index update failed | started=%s | finished=%s | errors=1 | duration=%.2fs",
            started_at.isoformat(timespec="seconds"),
            finished_at.isoformat(timespec="seconds"),
            duration,
        )
        raise


if __name__ == "__main__":
    main()
