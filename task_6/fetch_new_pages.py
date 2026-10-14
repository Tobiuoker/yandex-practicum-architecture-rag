from __future__ import annotations

import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from faker import Faker
from task_2.wiki_utils import character_alias
from task_2.wiki_utils import clean_wikitext
from task_2.wiki_utils import get_category_pages
from task_2.wiki_utils import get_wikitext
from task_2.wiki_utils import load_categories
from task_2.wiki_utils import make_fake_name
from task_2.wiki_utils import make_replacer
from task_6.logging_utils import setup_logging

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

KNOWLEDGE_BASE = PROJECT_ROOT / "task_2" / "knowledge_base"
CATEGORIES_FILE = PROJECT_ROOT / "task_2" / "categories.json"
TERMS_FILE = PROJECT_ROOT / "task_2" / "terms_map.json"
MANIFEST_FILE = PROJECT_ROOT / "task_2" / "manifest.json"
LOG_DIR = PROJECT_ROOT / "logs"

MAX_NEW_PAGES = int(os.getenv("FETCH_MAX_PAGES", "5"))
MIN_CHARS = 600
SEED = 42
DEPTH = 2


logger = logging.getLogger(__name__)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def get_next_doc_number() -> int:
    numbers = []
    for path in KNOWLEDGE_BASE.glob("doc_*.md"):
        match = re.fullmatch(r"doc_(\d+)\.md", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def discover_unseen_candidates(
    session: requests.Session,
    categories: list[dict],
    seen_titles: set[str],
) -> list[tuple[str, str]]:
    """Scan configured categories and exclude entities already in manifest."""
    candidates: dict[str, str] = {}
    successful_categories = 0

    for item in categories:
        category = item["category"]
        kind = item["kind"]
        logger.info("Scanning category: %s", category)

        try:
            titles = get_category_pages(
                session=session,
                category=category,
                depth=DEPTH,
            )
            successful_categories += 1
        except Exception:
            logger.exception("Failed to scan category: %s", category)
            continue

        for title in titles:
            if title.casefold() not in seen_titles:
                candidates.setdefault(title, kind)

    if successful_categories == 0:
        raise RuntimeError("All category scans failed.")

    return sorted(candidates.items(), key=lambda item: item[0].casefold())


def add_page_mapping(
    *,
    title: str,
    kind: str,
    mapping: dict[str, str],
    used_replacements: set[str],
    fake: Faker,
    rng: random.Random,
) -> None:
    if title in mapping:
        return

    replacement = make_fake_name(
        fake=fake,
        rng=rng,
        kind=kind,
        used=used_replacements,
    )
    mapping[title] = replacement

    if kind == "character":
        alias = character_alias(title)
        if alias:
            mapping.setdefault(alias, replacement.split()[0])


def fetch_pages(
    session: requests.Session,
    candidates: list[tuple[str, str]],
    mapping: dict[str, str],
    manifest: list[dict],
) -> tuple[int, int]:
    """Download and append at most MAX_NEW_PAGES valid documents."""

    random.Random(SEED).shuffle(candidates)

    seen_titles = {item["original_title"].casefold() for item in manifest if item.get("original_title")}
    used_replacements = {value.casefold() for value in mapping.values()}

    fake = Faker("en_US")
    Faker.seed(SEED)
    rng = random.Random(SEED)

    next_number = get_next_doc_number()
    added = 0
    errors = 0

    for requested_title, kind in candidates:
        if added >= MAX_NEW_PAGES:
            break

        try:
            result = get_wikitext(
                session=session,
                title=requested_title,
            )
            if not result:
                continue

            canonical_title, raw = result
            if canonical_title.casefold() in seen_titles:
                continue

            text = clean_wikitext(raw)
            if len(text) < MIN_CHARS:
                continue

            add_page_mapping(
                title=canonical_title,
                kind=kind,
                mapping=mapping,
                used_replacements=used_replacements,
                fake=fake,
                rng=rng,
            )
            replacer = make_replacer(mapping)

            filename = f"doc_{next_number:04d}.md"
            (KNOWLEDGE_BASE / filename).write_text(
                f"# {replacer(canonical_title)}\n\n{replacer(text)}\n",
                encoding="utf-8",
            )

            manifest.append(
                {
                    "file": filename,
                    "kind": kind,
                    "original_title": canonical_title,
                    "replacement": replacer(canonical_title),
                }
            )
            seen_titles.add(canonical_title.casefold())

            logger.info("Added page: %s -> %s", canonical_title, filename)
            next_number += 1
            added += 1
        except Exception:
            errors += 1
            logger.exception("Failed to add page: %s", requested_title)

    return added, errors


def main() -> None:
    setup_logging(LOG_DIR / "fetch_new_pages.log")
    started = datetime.now().astimezone()
    logger.info("Fetch started | time=%s", started.isoformat(timespec="seconds"))

    KNOWLEDGE_BASE.mkdir(parents=True, exist_ok=True)
    categories = load_categories(CATEGORIES_FILE)
    mapping = load_json(TERMS_FILE, {})
    manifest = load_json(MANIFEST_FILE, [])
    seen_titles = {item["original_title"].casefold() for item in manifest if item.get("original_title")}

    session = requests.Session()

    candidates = discover_unseen_candidates(session, categories, seen_titles)
    logger.info("Unseen candidates: %d", len(candidates))

    added, errors = fetch_pages(session, candidates, mapping, manifest)
    save_json(TERMS_FILE, mapping)
    save_json(MANIFEST_FILE, manifest)

    finished = datetime.now().astimezone()
    logger.info(
        "Fetch completed | started=%s | finished=%s | added=%d | errors=%d",
        started.isoformat(timespec="seconds"),
        finished.isoformat(timespec="seconds"),
        added,
        errors,
    )


if __name__ == "__main__":
    main()
