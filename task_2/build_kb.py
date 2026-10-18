from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests

from .wiki_utils import clean_wikitext
from .wiki_utils import generate_terms_map
from .wiki_utils import get_category_pages
from .wiki_utils import get_wikitext
from .wiki_utils import load_categories
from .wiki_utils import make_replacer

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TASK_DIR
DEFAULT_CATEGORIES_FILE = TASK_DIR / "categories.json"


def discover_candidates(
    session: requests.Session,
    categories: list[dict],
    *,
    depth: int,
) -> dict[str, str]:
    """Collect candidate titles and keep the first detected entity type."""
    candidates: dict[str, str] = {}

    for item in categories:
        category = item["category"]
        kind = item["kind"]
        print(f"Scanning {category}...")

        try:
            titles = get_category_pages(
                session=session,
                category=category,
                depth=depth,
            )
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue

        print(f"  found: {len(titles)}")
        for title in titles:
            candidates.setdefault(title, kind)

    if not candidates:
        raise RuntimeError("No candidate pages were found.")

    return candidates


def download_articles(
    session: requests.Session,
    candidates: dict[str, str],
    *,
    pages: int,
    min_chars: int,
    seed: int,
) -> list[dict]:
    """Download and clean a bounded set of pages before anonymization."""
    items = list(candidates.items())
    random.Random(seed).shuffle(items)

    articles: list[dict] = []
    seen_titles: set[str] = set()

    for requested_title, kind in items:
        if len(articles) >= pages:
            break

        print(f"[{len(articles) + 1}/{pages}] {requested_title}")

        try:
            result = get_wikitext(
                session=session,
                title=requested_title,
            )
            if not result:
                print("  skip: no content")
                continue

            canonical_title, raw = result
            canonical_key = canonical_title.casefold()
            if canonical_key in seen_titles:
                print("  skip: duplicate redirect target")
                continue

            text = clean_wikitext(raw)
        except Exception as exc:
            print(f"  error: {exc}")
            continue

        if len(text) < min_chars:
            print(f"  skip: too short ({len(text)} chars)")
            continue

        seen_titles.add(canonical_key)
        articles.append(
            {
                "title": canonical_title,
                "kind": kind,
                "text": text,
            }
        )
        print("  ok")

    if not articles:
        raise RuntimeError("No suitable articles were downloaded.")

    return articles


def write_knowledge_base(
    articles: list[dict],
    *,
    output_dir: Path,
    seed: int,
) -> None:
    """Generate terms_map.json and write anonymized Markdown documents."""
    knowledge_base = output_dir / "knowledge_base"
    knowledge_base.mkdir(parents=True, exist_ok=True)

    for old_file in knowledge_base.glob("*.md"):
        old_file.unlink()

    mapping = generate_terms_map(entities=articles, seed=seed)
    replacer = make_replacer(mapping)

    (output_dir / "terms_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = []
    for index, article in enumerate(articles, start=1):
        new_title = replacer(article["title"])
        new_text = replacer(article["text"])
        filename = f"doc_{index:04d}.md"

        (knowledge_base / filename).write_text(
            f"# {new_title}\n\n{new_text}\n",
            encoding="utf-8",
        )

        manifest.append(
            {
                "file": filename,
                "kind": article["kind"],
                "original_title": article["title"],
                "replacement": new_title,
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Documents:     {len(articles)}")
    print(f"Replacements:  {len(mapping)}")
    print(f"Knowledge DB:  {knowledge_base}")
    print("Index into RAG only: knowledge_base/*.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES_FILE)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = load_categories(args.categories)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "WitcherRAGDataset/1.0 (educational project)",
        }
    )

    candidates = discover_candidates(
        session,
        categories,
        depth=args.depth,
    )
    articles = download_articles(
        session,
        candidates,
        pages=args.pages,
        min_chars=args.min_chars,
        seed=args.seed,
    )
    write_knowledge_base(
        articles,
        output_dir=args.out.resolve(),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
