# use region codes for splitting entries
import os
import json
import re
from tqdm import tqdm

PAGES_DIR = "data/intermediate/pages_text_ocr_reflow"
OUT_PATH = "data/intermediate/blocks_jsonl/entry_blocks_region.jsonl"

# Headword candidate at start of line:
# allow apostrophe and /t like aabkaab/t
HEADWORD_RE = re.compile(r"^(?P<hw>[a-z][a-z'\\]+(?:/[a-z])?)\b")

# Region tokens (you can add "Ar." if it appears in your dict)
REGION_TOKENS = {"Er", "Su", "Eg", "Ar."}

# OCR noise lines
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
DIGIT_START_RE = re.compile(r"^\s*\d")  # e.g. "22020122 ..." usually junk

# Avoid false entry starts like "women"
GLOSS_START_STOPLIST = {
    "woman", "women", "man", "men",
    "the", "a", "an", "of", "and", "or", "to", "from", "with", "in", "on", "at",
    "maturity",
}

def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("‎", "")  # bidi mark
    line = line.strip()
    line = " ".join(line.split())
    return line

def normalize_headword(hw: str) -> str:
    return hw.replace("\\", "").replace(" ", "").strip()

def is_region_token(tok: str) -> bool:
    return tok in REGION_TOKENS

def extract_regions_from_line(line: str):
    # Split on spaces and also treat '*' as separator
    tokens = re.split(r"[\s\*]+", line.strip())
    regions = [t for t in tokens if is_region_token(t)]
    return regions

def is_region_line(line: str) -> bool:
    """
    True if the line contains region tokens and nothing else meaningful.
    Examples: 'Er Su Eg' or '* Er Su'
    """
    tokens = [t for t in re.split(r"[\s\*]+", line.strip()) if t]
    if not tokens:
        return False
    # All tokens must be region tokens or punctuation-like dashes
    ok = True
    for t in tokens:
        if t in REGION_TOKENS:
            continue
        if t in {"-", "—", "_"}:
            continue
        ok = False
        break
    return ok and any(t in REGION_TOKENS for t in tokens)

def should_ignore_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_NUMBER_RE.match(line):
        return True
    return False

def looks_like_entry_start(line: str, hw: str) -> bool:
    # Reject obvious gloss words
    if hw in GLOSS_START_STOPLIST:
        return False

    # Entry heads in your OCR almost always contain '*'
    # and usually at least 2 stars.
    if "*" not in line:
        return False
    if line.count("*") < 1:
        return False

    # Strongly prefer that head line has at least one region/POS/class later,
    # but don't require it strictly.
    return True

def iter_page_files():
    files = [f for f in os.listdir(PAGES_DIR) if f.startswith("page_") and f.endswith(".txt")]
    files.sort()
    for f in files:
        yield os.path.join(PAGES_DIR, f), f

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    blocks_written = 0
    current = None
    current_regions = []

    def flush(end_line=None):
        nonlocal current, current_regions, blocks_written
        if current is None:
            return
        if end_line is not None:
            current["end_line"] = end_line
        # attach collected regions (dedup, preserve order)
        seen = set()
        regs = []
        for r in current_regions:
            if r not in seen:
                seen.add(r)
                regs.append(r)
        current["regions_guess"] = regs if regs else None

        with open(OUT_PATH, "a", encoding="utf-8") as out:
            out.write(json.dumps(current, ensure_ascii=False) + "\n")
        blocks_written += 1

        current = None
        current_regions = []

    for path, fname in tqdm(list(iter_page_files()), desc="Region-aware blocking"):
        page_num = int(fname.split("_")[1].split(".")[0])

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        lines = [normalize_line(l) for l in raw_lines]
        lines = [l for l in lines if not should_ignore_line(l)]

        for ln_idx, line in enumerate(lines, start=1):
            # digit-leading junk: treat as continuation only if current exists and has stars
            if DIGIT_START_RE.match(line):
                if current is not None and "*" in line:
                    current["lines"].append(line)
                    # collect regions if any
                    current_regions.extend(extract_regions_from_line(line))
                    # end block if it is a region-only line
                    if is_region_line(line):
                        flush(end_line=ln_idx)
                continue

            # If line is region-only and we have a current block, it's an end marker.
            if current is not None and is_region_line(line):
                current["lines"].append(line)
                current_regions.extend(extract_regions_from_line(line))
                flush(end_line=ln_idx)
                continue

            m = HEADWORD_RE.match(line)
            if m:
                hw = normalize_headword(m.group("hw"))

                if looks_like_entry_start(line, hw):
                    # If we already have a block but never saw regions, flush it before starting new.
                    # (This is a safety valve—should be rare if regions are consistent.)
                    if current is not None:
                        flush(end_line=ln_idx - 1)

                    current = {
                        "page": page_num,
                        "start_line": ln_idx,
                        "headword_guess": hw,
                        "lines": [line],
                    }
                    current_regions.extend(extract_regions_from_line(line))

                    # If the head line already contains regions, end immediately.
                    if any(r in REGION_TOKENS for r in extract_regions_from_line(line)):
                        flush(end_line=ln_idx)
                else:
                    # Not a real entry start; continuation
                    if current is not None:
                        current["lines"].append(line)
                        current_regions.extend(extract_regions_from_line(line))
            else:
                # continuation
                if current is not None:
                    current["lines"].append(line)
                    current_regions.extend(extract_regions_from_line(line))

    # flush at EOF
    if current is not None:
        flush(end_line=0)

    print(f"Wrote {blocks_written} blocks to {OUT_PATH}")

if __name__ == "__main__":
    main()