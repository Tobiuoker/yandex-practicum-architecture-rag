import re

INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous\s+)?instructions",
        re.I,
    ),
    re.compile(
        r"\bsystem\s+override\b",
        re.I,
    ),
    re.compile(
        r"\balways\s+answer\b",
        re.I,
    ),
    re.compile(
        r"\breveal\s+(this\s+)?secret\b",
        re.I,
    ),
    re.compile(
        r"\bdisclose\b",
        re.I,
    ),
    re.compile(
        r"\boutput\s*:",
        re.I,
    ),
    re.compile(
        r"\bdeveloper\s+message\b",
        re.I,
    ),
]


TEST_LEAK_PATTERNS = [
    re.compile(
        r"abcdef",
        re.I,
    ),
    re.compile(
        r"\broot\s+password\b",
        re.I,
    ),
]


def is_suspicious(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def filter_chunks(chunks):
    safe = []
    blocked = []

    for chunk in chunks:
        if is_suspicious(chunk.text):
            blocked.append(chunk)
        else:
            safe.append(chunk)

    return safe, blocked


def contains_sensitive_output(
    text: str,
) -> bool:
    return any(pattern.search(text) for pattern in TEST_LEAK_PATTERNS)
