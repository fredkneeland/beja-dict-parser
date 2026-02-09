# first try with ocr results

import os
import json
import re

# Set these paths
INPUT_TXT = "data/input/simple_copy.txt"
OUT_JSONL = "data/intermediate/blocks_jsonl/entry_blocks_from_txt.jsonl"

# A Beja headword line normally starts with a lowercase latin token
HEADWORD_START_RE = re.compile(r"^(?P<hw>[a-z][a-z']{1,})\b")

# Region tokens that may appear alone on lines
REGION_TOKENS = {"Er", "Su", "Eg", "Ar."}

# English gloss words that should not be treated as headwords if they appear at line start
GLOSS_STOPLIST = {
    "women", "woman", "men", "man", "maturity",
    "the", "a", "an", "of", "and", "or", "to", "from", "with", "in", "on", "at"
}

def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").strip()
    line = " ".join(line.split())
    return line

def looks_like_region_line(line: str) -> bool:
    toks = set(line.split())
    return bool(toks) and toks.issubset(REGION_TOKENS)

def split_stuck_region_prefix(line: str) -> str:
    """
    Fix cases like: 'Er Su Egaareeb atee ...'
    If line starts with region tokens and then immediately a lemma without space/newline separation,
    try to separate: 'Er Su Eg' + rest.
    """
    # Handle common stuck: "Egaareeb" where "Eg" is glued to lemma
    line = re.sub(r"\bEg(?=[a-z])", "Eg ", line)
    return line

def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    # Normalize and drop empties
    lines = [normalize_line(l) for l in raw_lines]
    lines = [l for l in lines if l]

    blocks = []
    current = None

    def flush():
        nonlocal current
        if current is not None:
            blocks.append(current)
            current = None

    i = 0
    while i < len(lines):
        line = split_stuck_region_prefix(lines[i])

        # Region-only lines attach to current block
        if looks_like_region_line(line):
            if current is not None:
                current["lines"].append(line)
            i += 1
            continue

        # Detect a new entry start
        m = HEADWORD_START_RE.match(line)
        if m:
            hw = m.group("hw")

            # Avoid common gloss words accidentally at line start
            if hw in GLOSS_STOPLIST:
                # treat as continuation
                if current is not None:
                    current["lines"].append(line)
                i += 1
                continue

            # Start new block
            flush()
            current = {
                "source": {"type": "copypaste_txt", "line_start": i + 1},
                "headword_guess": hw,
                "lines": [line],
            }
            i += 1
            continue

        # Otherwise it's continuation
        if current is not None:
            current["lines"].append(line)
        i += 1

    flush()

    # Write JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for b in blocks:
            out.write(json.dumps(b, ensure_ascii=False) + "\n")

    print(f"Wrote {len(blocks)} blocks to {OUT_JSONL}")

if __name__ == "__main__":
    main()