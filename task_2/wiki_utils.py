from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from collections import deque
from pathlib import Path

import mwparserfromhell
import requests
from bs4 import BeautifulSoup
from faker import Faker

API_URL = "https://witcher.fandom.com/api.php"

BAD_TITLE_PARTS = (
    "(disambiguation)",
    "/gallery",
    "/netflix",
    "/hexer",
    "(gwent card)",
    "(card)",
    "(quest)",
    "walkthrough",
    "journal entry",
)

SKIP_SECTIONS = (
    "references",
    "external links",
    "notes",
    "sources",
    "bibliography",
    "gallery",
    "trivia",
)

# Common concepts must not be anonymized as named entities.
GENERIC_TITLES = {
    "alchemy",
    "armor",
    "battle",
    "beast",
    "beasts",
    "character",
    "characters",
    "city",
    "cities",
    "creature",
    "creatures",
    "dwarf",
    "dwarves",
    "elf",
    "elves",
    "empire",
    "event",
    "events",
    "human",
    "humans",
    "king",
    "kingdom",
    "kingdoms",
    "location",
    "locations",
    "mage",
    "mages",
    "magic",
    "monster",
    "monsters",
    "queen",
    "race",
    "races",
    "region",
    "regions",
    "school",
    "settlement",
    "settlements",
    "sword",
    "swords",
    "town",
    "towns",
    "village",
    "villages",
    "war",
    "weapon",
    "weapons",
    "witcher",
    "witchers",
}

LOCATION_SUFFIXES = [
    "Reach",
    "Vale",
    "Hold",
    "Haven",
    "March",
    "Moor",
    "Coast",
    "Fen",
]

CREATURE_SUFFIXES = [
    "Wraith",
    "Crawler",
    "Stalker",
    "Fiend",
    "Beast",
    "Watcher",
    "Howler",
    "Spawn",
]

ITEM_SUFFIXES = [
    "Relic",
    "Blade",
    "Sigil",
    "Talisman",
    "Artifact",
    "Stone",
    "Charm",
    "Token",
]

OTHER_SUFFIXES = [
    "Order",
    "Pact",
    "Circle",
    "Accord",
    "Chronicle",
    "Conclave",
    "Covenant",
    "League",
]

TITLE_PREFIXES_TO_IGNORE = {
    "king",
    "queen",
    "prince",
    "princess",
    "emperor",
    "empress",
    "sir",
    "lord",
    "lady",
    "baron",
    "count",
    "duke",
    "general",
    "captain",
    "master",
}


def api_request(session: requests.Session, **params) -> dict:
    params.update(
        {
            "format": "json",
            "formatversion": 2,
        }
    )

    response = session.get(
        API_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    time.sleep(0.2)

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data


def valid_title(title: str) -> bool:
    lowered = title.casefold().strip()

    if not lowered:
        return False

    if "/" in title:
        return False

    if lowered.startswith("list of "):
        return False

    return not any(part in lowered for part in BAD_TITLE_PARTS)


def is_anonymizable_entity(title: str) -> bool:
    """
    Only named entities should be anonymized.

    Common concepts such as "city", "human" or "witcher" are left intact,
    because replacing them corrupts the meaning of sentences.
    """

    normalized = re.sub(
        r"\s*\([^)]*\)\s*$",
        "",
        title,
    ).strip()

    if not normalized:
        return False

    return normalized.casefold() not in GENERIC_TITLES


def get_category_pages(
    session: requests.Session,
    category: str,
    depth: int = 2,
) -> list[str]:
    queue = deque([(category, 0)])
    visited_categories = set()
    pages = set()

    while queue:
        current_category, current_depth = queue.popleft()

        if current_category in visited_categories:
            continue

        visited_categories.add(current_category)
        continuation = None

        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": current_category,
                "cmtype": "page|subcat",
                "cmlimit": "max",
            }

            if continuation:
                params["cmcontinue"] = continuation

            data = api_request(session, **params)
            members = data.get("query", {}).get("categorymembers", [])

            for item in members:
                title = item["title"]
                namespace = item["ns"]

                if namespace == 0 and valid_title(title):
                    pages.add(title)

                elif namespace == 14 and current_depth < depth:
                    queue.append((title, current_depth + 1))

            continuation = data.get("continue", {}).get("cmcontinue")

            if not continuation:
                break

    return sorted(pages)


def get_wikitext(
    session: requests.Session,
    title: str,
) -> tuple[str, str] | None:
    data = api_request(
        session,
        action="parse",
        page=title,
        prop="text",
        redirects=1,
    )

    parsed = data.get("parse")

    if not parsed:
        return None

    soup = BeautifulSoup(
        parsed["text"],
        "html.parser",
    )

    root = soup.select_one(".mw-parser-output") or soup

    # Remove citations and service elements,
    # but keep visible link text.
    for element in root.select(
        """
        sup.reference,
        .mw-references-wrap,
        ol.references,
        script,
        style,
        .mw-editsection
        """
    ):
        element.decompose()

    text = root.get_text(
        "\n",
        strip=True,
    )

    return (
        parsed["title"],
        text,
    )


def clean_wikitext(raw: str) -> str:
    code = mwparserfromhell.parse(raw)

    # Remove references together with their contents.
    for tag in code.filter_tags():
        if str(tag.tag).strip().lower() == "ref":
            code.remove(tag)

    return code.strip_code(
        normalize=True,
        collapse=True,
    ).strip()


def _entity_seed(original: str, seed: int, salt: int) -> int:
    digest = hashlib.sha256(f"{seed}|{salt}|{original}".encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


def _generate_name_candidate(
    fake: Faker,
    rng: random.Random,
    kind: str,
) -> str:
    if kind == "character":
        return f"{fake.first_name()} {fake.last_name()}"
    if kind == "location":
        return f"{fake.last_name()} {rng.choice(LOCATION_SUFFIXES)}"
    if kind == "creature":
        return f"{fake.last_name()} {rng.choice(CREATURE_SUFFIXES)}"
    if kind == "item":
        return f"{fake.last_name()} {rng.choice(ITEM_SUFFIXES)}"
    return f"{fake.last_name()} {rng.choice(OTHER_SUFFIXES)}"


def generate_replacement(
    *,
    original: str,
    kind: str,
    used: set[str],
    seed: int = 42,
) -> str:
    """
    Generate a deterministic fake name for an entity.

    The replacement is stable for the same original name, even when
    discovery order changes between runs.
    """

    salt = 0

    while True:
        entity_seed = _entity_seed(
            original,
            seed,
            salt,
        )

        fake = Faker("en_US")
        fake.seed_instance(entity_seed)

        rng = random.Random(entity_seed)

        value = _generate_name_candidate(fake, rng, kind)

        key = value.casefold()

        if key not in used:
            used.add(key)
            return value

        salt += 1


def make_fake_name(
    fake: Faker,
    rng: random.Random,
    kind: str,
    used: set[str],
) -> str:
    """
    Compatibility helper for older update scripts.
    New code should prefer generate_replacement().
    """

    while True:
        value = _generate_name_candidate(fake, rng, kind)

        key = value.casefold()

        if key not in used:
            used.add(key)
            return value


def character_alias(title: str) -> str | None:
    title = re.sub(
        r"\s*\([^)]*\)\s*$",
        "",
        title,
    ).strip()

    words = title.split()

    if len(words) < 2:
        return None

    first = words[0].strip(".,'\"")

    if first.casefold() in TITLE_PREFIXES_TO_IGNORE:
        return None

    if len(first) < 4:
        return None

    if not re.fullmatch(
        r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+",
        first,
    ):
        return None

    return first


def generate_terms_map(
    entities: list[dict],
    seed: int = 42,
) -> dict[str, str]:
    """
    Build a mapping from a broader entity catalog, not just the final corpus.

    This reduces leakage of known proper names that are mentioned inside
    selected articles but whose own pages were not selected.
    """

    normalized_entities = []
    seen_titles = set()

    for entity in entities:
        title = entity["title"].strip()
        kind = entity["kind"]
        key = title.casefold()

        if key in seen_titles:
            continue

        if not is_anonymizable_entity(title):
            continue

        seen_titles.add(key)

        normalized_entities.append(
            {
                "title": title,
                "kind": kind,
            }
        )

    normalized_entities.sort(
        key=lambda item: (
            item["kind"],
            item["title"].casefold(),
        )
    )

    mapping: dict[str, str] = {}
    used_replacements: set[str] = set()

    for entity in normalized_entities:
        title = entity["title"]

        mapping[title] = generate_replacement(
            original=title,
            kind=entity["kind"],
            used=used_replacements,
            seed=seed,
        )

    # Add first-name aliases only when they are unique.
    aliases = []

    for entity in normalized_entities:
        if entity["kind"] != "character":
            continue

        alias = character_alias(entity["title"])

        if alias:
            aliases.append((alias, entity["title"]))

    alias_counts = Counter(alias.casefold() for alias, _ in aliases)

    full_titles = {entity["title"].casefold() for entity in normalized_entities}

    for alias, full_title in aliases:
        if alias_counts[alias.casefold()] != 1:
            continue

        if alias.casefold() in full_titles:
            continue

        mapping.setdefault(
            alias,
            mapping[full_title].split()[0],
        )

    return mapping


def make_replacer(mapping: dict[str, str]):
    if not mapping:
        return lambda text: text

    lookup = {original.casefold(): replacement for original, replacement in mapping.items()}

    terms = sorted(
        mapping,
        key=len,
        reverse=True,
    )

    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms) + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    def replace(text: str) -> str:
        return pattern.sub(
            lambda match: lookup[match.group(0).casefold()],
            text,
        )

    return replace


def load_categories(path: Path) -> list[dict]:
    """Load category configuration from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("categories.json must contain a JSON array")
    return data
