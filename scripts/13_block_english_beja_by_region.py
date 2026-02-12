import os
import json
import re
from tqdm import tqdm

INPUT_DIR = "data/intermediate/engbeja_pages_text_2col"
OUT_PATH = "data/intermediate/english_beja_blocks.jsonl"

REGIONS = {"Er", "Su", "Eg"}

# Matches things like: "33", "33.", "(33)", "٣٣", "33‏"
LINE_NUMBER_RE = re.compile(r"^\s*[\(\[\{]?\s*[0-9٠-٩]{1,4}\s*[\)\]\}]?\s*[\.\:\-–—]?\s*$")
LEADING_NUMBER_RE = re.compile(r"^\s*[0-9٠-٩]{1,4}\s+(.*)$")

LATIN_LETTERS_RE = re.compile(r"[A-Za-z]")
ARABIC_LETTERS_RE = re.compile(r"[\u0600-\u06FF]")
DIGITS_RE = re.compile(r"[0-9٠-٩]")
NON_DIGIT_RE = re.compile(r"[^0-9٠-٩]")

def normalize(line: str) -> str:
    # remove bidi/zero-width junk that often attaches to numbers
    line = line.replace("‎", "").replace("\u200f", "").replace("\u200e", "")
    line = line.replace("\u202a", "").replace("\u202b", "").replace("\u202c", "")
    line = line.replace("\ufeff", "")
    return " ".join(line.split()).strip()

def is_header_or_comment(line: str) -> bool:
    return (not line) or line.startswith("#")

def is_line_number(line: str) -> bool:
    """
    Drop short OCR junk like:
      '33', '(33)', ')2', '-- ٣', '— 12', '2)'
    i.e. contains no letters and is basically just digits + punctuation.
    """
    s = normalize(line)
    if not s:
        return True

    # If it contains any Latin or Arabic letters, it's not a number/junk line
    if LATIN_LETTERS_RE.search(s) or ARABIC_LETTERS_RE.search(s):
        return False

    # Must contain at least one digit
    if not DIGITS_RE.search(s):
        return False

    # Remove digits; if remaining chars are only punctuation/space, it's junk
    rest = re.sub(r"[0-9٠-٩]", "", s)
    rest = rest.strip().strip("()[]{}.,;:!?'\"+-–—_/\\|<>")

    # Also keep it short to avoid deleting something weird but meaningful
    return rest == "" and len(s) <= 6

def strip_leading_number(line: str) -> str:
    m = LEADING_NUMBER_RE.match(line)
    return m.group(1).strip() if m else line

def tokens(line: str):
    return [t for t in re.split(r"[\s\*]+", line) if t]

def has_region_token(line: str) -> bool:
    return any(t in REGIONS for t in tokens(line))

def is_region_only_line(line: str) -> bool:
    ts = tokens(line)
    return bool(ts) and all(t in REGIONS for t in ts)

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))

    current = None
    last_flushed = None
    blocks_written = 0

    def write_entry(entry):
        nonlocal blocks_written
        # Final safety: drop a leading pure number line if it slipped in
        while entry["lines"] and is_line_number(normalize(entry["lines"][0])):
            entry["lines"].pop(0)
            entry["start_line"] = entry["start_line"] + 1

        if not entry["lines"]:
            return

        with open(OUT_PATH, "a", encoding="utf-8") as out:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        blocks_written += 1

    def flush(end_line: int):
        nonlocal current, last_flushed
        if not current:
            return
        current["end_line"] = end_line
        last_flushed = current
        current = None

    def commit_last_flushed():
        nonlocal last_flushed
        if last_flushed:
            write_entry(last_flushed)
            last_flushed = None

    for fname in tqdm(files, desc="Blocking English→Beja v3"):
        page = int(fname.split("_")[1].split(".")[0])
        path = os.path.join(INPUT_DIR, fname)

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        lines = [normalize(l) for l in raw_lines]
        lines = [l for l in lines if not is_header_or_comment(l)]

        for i, line in enumerate(lines, start=1):
            # EARLIEST filter: drop standalone line numbers
            if is_line_number(line):
                continue

            # If we have a pending flushed entry, decide whether to attach region-only wrap
            if last_flushed is not None:
                if is_region_only_line(line):
                    last_flushed["lines"].append(line)
                    last_flushed["end_line"] = i
                    write_entry(last_flushed)
                    last_flushed = None
                    continue
                else:
                    commit_last_flushed()

            # Region-only line ends the current entry
            if is_region_only_line(line):
                if current is not None:
                    current["lines"].append(line)
                    flush(i)
                continue

            # If starting a new entry and line begins with a number, strip it (e.g. "33 accountant * ...")
            if current is None:
                line = strip_leading_number(line)
                if not line or is_line_number(line):
                    continue
                current = {"page": page, "start_line": i, "lines": [line]}
            else:
                current["lines"].append(line)

            # Region token ends entry (but keep it pending to allow next-line region-only continuation)
            if has_region_token(line):
                flush(i)

        # End of page: commit pending
        if last_flushed is not None:
            write_entry(last_flushed)
            last_flushed = None

        if current is not None:
            current["end_line"] = 0
            write_entry(current)
            current = None

    print(f"Wrote {blocks_written} blocks → {OUT_PATH}")

if __name__ == "__main__":
    main()