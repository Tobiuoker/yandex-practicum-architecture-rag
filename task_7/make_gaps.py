from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = BASE_DIR.parent

KB_DIR = PROJECT_ROOT / "task_2" / "knowledge_base"

REMOVED_DIR = BASE_DIR / "removed"

GAPS_FILE = BASE_DIR / "gaps.json"

STATE_FILE = BASE_DIR / "gap_state.json"


def load_gaps() -> list[str]:
    return json.loads(GAPS_FILE.read_text(encoding="utf-8"))


def find_files(
    entities: list[str],
) -> list[Path]:
    """
    Remove every document containing the entity,
    not only the entity's primary article.

    This creates a real knowledge gap.
    """

    result = set()

    for file in KB_DIR.glob("*.md"):
        text = file.read_text(encoding="utf-8").casefold()

        if any(entity.casefold() in text for entity in entities):
            result.add(file)

    return sorted(result)


def apply():
    REMOVED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    entities = load_gaps()
    files = find_files(entities)

    state = []

    for file in files:
        target = REMOVED_DIR / file.name

        shutil.move(
            file,
            target,
        )

        state.append(
            {
                "file": file.name,
                "original": str(file),
                "backup": str(target),
            }
        )

        print(f"Removed: {file.name}")

    STATE_FILE.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nDocuments removed: " f"{len(files)}")


def restore():
    if not STATE_FILE.exists():
        print("Nothing to restore")
        return

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))

    for item in state:
        source = Path(item["backup"])

        target = Path(item["original"])

        if source.exists():
            shutil.move(
                source,
                target,
            )

            print(f"Restored: " f"{target.name}")

    STATE_FILE.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=[
            "apply",
            "restore",
        ],
    )

    args = parser.parse_args()

    if args.action == "apply":
        apply()
    else:
        restore()


if __name__ == "__main__":
    main()
