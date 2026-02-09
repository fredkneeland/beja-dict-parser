# second try at making blocks using region markers
import os
import json
import re
from tqdm import tqdm

PAGES_DIR = "data/intermediate/pages_text_ocr_reflow"
OUT_PATH = "data/intermediate/blocks_jsonl/entry_blocks_region2.jsonl"

# --- Patterns / token sets ---
HEADWORD_AT_START = re.compile(r"^(?P<hw>[a-z][a-z'\\]+(?:/[a-z])?)\b")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

REGION_TOKENS = {"Er", "Su", "Eg", "Ar."}
CLASS_TOKENS = {"Sem", "Cush"}
POS_TOKENS = {"N", "V", "Adj", "Adv", "Pron", "Phr", "Intj"}

# Common continuation tokens that should NEVER be treated as headwords
NEVER_HEADWORDS = {
    "sg", "pl", "pl.", "m", "f", "mf", "m.", "f.",
    "-", "—", "_"
}

# OCR often prefixes garbage digits before a real token
LEADING_DIGITS_RE = re.compile(r"^\s*\d+\s*(.*)$")

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

# English stoplist when OCR moves gloss words to line start
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

def extract_regions_from_line(line: str):
    tokens = re.split(r"[\s\*]+", line.strip())
    return [t for t in tokens if t in REGION_TOKENS]

def is_region_line(line: str) -> bool:
    """
    True if line is basically just region tokens (+ optional dashes).
    """
    tokens = [t for t in re.split(r"[\s\*]+", line.strip()) if t]
    if not tokens:
        return False
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

def looks_like_entry_head_line(line: str) -> bool:
    """
    Decide if a line 'looks like' a dictionary entry head.
    Requirements:
    - contains '*' (dictionary structure)
    - contains at least 2 star-separated fields OR has POS/class/region markers
    """
    if "*" not in line:
        return False
    if line.count("*") >= 2:
        return True

    # fallback: if it contains markers, treat as entry-like
    tokens = set(re.split(r"[\s\*]+", line))
    if tokens.intersection(POS_TOKENS | CLASS_TOKENS | REGION_TOKENS):
        return True
    return False

def strip_leading_digits(line: str) -> str:
    m = LEADING_DIGITS_RE.match(line)
    return m.group(1).strip() if m else line

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
    last_good_headword = None  # used to repair lines like "2203 daatiya ..." -> "aada daatiya ..."

    def flush(end_line=None):
        nonlocal current, current_regions, blocks_written
        if current is None:
            return

        if end_line is not None:
            current["end_line"] = end_line

        # attach regions guess
        regs = []
        seen = set()
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

    for path, fname in tqdm(list(iter_page_files()), desc="Region-aware blocking v2"):
        page_num = int(fname.split("_")[1].split(".")[0])

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        lines = [normalize_line(l) for l in raw_lines]
        lines = [l for l in lines if not should_ignore_line(l)]

        for ln_idx, raw in enumerate(lines, start=1):
            line = raw

            # --- Fix #2: sg/pl/etc. lines are never entry starts ---
            first_tok = line.split()[0] if line.split() else ""
            if first_tok in NEVER_HEADWORDS:
                if current is not None:
                    current["lines"].append(line)
                    current_regions.extend(extract_regions_from_line(line))
                    if is_region_line(line):
                        flush(end_line=ln_idx)
                continue

            # --- Fix #1 + #3: handle digit-prefixed lines like "2203 daatiya * ..." ---
            if line and line[0].isdigit():
                stripped = strip_leading_digits(line)

                # If after stripping digits it looks like an entry head, treat it as new entry.
                if looks_like_entry_head_line(stripped):
                    # Try to repair: if it starts with a token like 'daatiya' (not a real lemma start),
                    # prepend last_good_headword if available.
                    m_hw = HEADWORD_AT_START.match(stripped)
                    if m_hw:
                        hw = normalize_headword(m_hw.group("hw"))
                        # If the "headword" is a common gloss word, don't start a new entry.
                        if hw in GLOSS_START_STOPLIST or hw in NEVER_HEADWORDS:
                            # continuation
                            if current is not None:
                                current["lines"].append(line)
                                current_regions.extend(extract_regions_from_line(line))
                            continue

                        # Start new entry normally
                        if current is not None:
                            flush(end_line=ln_idx - 1)

                        current = {"page": page_num, "start_line": ln_idx, "headword_guess": hw, "lines": [stripped]}
                        current_regions.extend(extract_regions_from_line(stripped))
                        last_good_headword = hw

                        if is_region_line(stripped) or extract_regions_from_line(stripped):
                            # if somehow it already includes region-only line, flush
                            pass
                        continue

                    else:
                        # No headword at start after stripping digits.
                        # If it begins with something like "daatiya * ..." and we have last_good_headword,
                        # then repair to "last_good_headword <tail> ..."
                        if last_good_headword:
                            repaired = f"{last_good_headword} {stripped}"
                            # Flush current before starting this new derived entry
                            if current is not None:
                                flush(end_line=ln_idx - 1)
                            current = {
                                "page": page_num,
                                "start_line": ln_idx,
                                "headword_guess": last_good_headword,
                                "lines": [repaired],
                            }
                            current_regions.extend(extract_regions_from_line(repaired))
                            continue

                # Otherwise treat digit line as continuation/junk
                if current is not None and "*" in stripped:
                    current["lines"].append(stripped)
                    current_regions.extend(extract_regions_from_line(stripped))
                    if is_region_line(stripped):
                        flush(end_line=ln_idx)
                continue

            # --- Normal region-only line ends the current entry ---
            if current is not None and is_region_line(line):
                current["lines"].append(line)
                current_regions.extend(extract_regions_from_line(line))
                flush(end_line=ln_idx)
                continue

            # --- Normal headword line at start ---
            m = HEADWORD_AT_START.match(line)
            if m:
                hw = normalize_headword(m.group("hw"))

                # Don't treat sg/pl/etc. or common gloss words as headwords
                if hw in NEVER_HEADWORDS or hw in GLOSS_START_STOPLIST:
                    if current is not None:
                        current["lines"].append(line)
                        current_regions.extend(extract_regions_from_line(line))
                    continue

                # Require dictionary structure
                if looks_like_entry_head_line(line):
                    # If we already have an entry open, flush it (even if missing region)
                    if current is not None:
                        flush(end_line=ln_idx - 1)

                    current = {"page": page_num, "start_line": ln_idx, "headword_guess": hw, "lines": [line]}
                    current_regions.extend(extract_regions_from_line(line))
                    last_good_headword = hw

                    # If the head line already includes regions AND ends cleanly, we can flush later when next starts.
                    continue
                else:
                    # continuation
                    if current is not None:
                        current["lines"].append(line)
                        current_regions.extend(extract_regions_from_line(line))
                    continue

            # --- Continuation line ---
            if current is not None:
                current["lines"].append(line)
                current_regions.extend(extract_regions_from_line(line))

                # Extra safety: if a continuation line contains region tokens and looks like it's the final tag line, flush.
                if is_region_line(line):
                    flush(end_line=ln_idx)

    # flush last entry
    if current is not None:
        flush(end_line=0)

    print(f"Wrote {blocks_written} blocks to {OUT_PATH}")

if __name__ == "__main__":
    main()