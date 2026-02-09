# 3rd try to fix multiple word entries and page ends not coming in correctly
import os
import json
import re
from tqdm import tqdm
from typing import Optional

PAGES_DIR = "data/intermediate/pages_text_ocr_reflow"
OUT_PATH = "data/intermediate/blocks_jsonl/entry_blocks_region3.jsonl"

REGION_TOKENS = {"Er", "Su", "Eg", "Ar."}
CLASS_TOKENS = {"Sem", "Cush"}
POS_TOKENS = {"N", "V", "Adj", "Adv", "Pron", "Phr", "Intj"}

NEVER_HEADWORDS = {"sg", "pl", "pl.", "m", "f", "mf", "m.", "f.", "-", "—", "_"}
GLOSS_START_STOPLIST = {
    "woman", "women", "man", "men",
    "the", "a", "an", "of", "and", "or", "to", "from", "with", "in", "on", "at",
    "maturity",
}

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
LEADING_DIGITS_RE = re.compile(r"^\s*\d+\s*(.*)$")

# Beja-ish token: lowercase latin with optional apostrophe or /t
BEJA_TOKEN_RE = re.compile(r"^[a-z][a-z']*(?:/[a-z])?$")

def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("‎", "")  # bidi mark
    line = line.strip()
    line = " ".join(line.split())
    return line

def strip_leading_digits(line: str) -> str:
    m = LEADING_DIGITS_RE.match(line)
    return m.group(1).strip() if m else line

def should_ignore_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_NUMBER_RE.match(line):
        return True
    return False

def extract_regions_from_line(line: str):
    tokens = re.split(r"[\s\*]+", line.strip())
    return [t for t in tokens if t in REGION_TOKENS]

def is_region_line(line: str) -> bool:
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

def looks_like_entry_head_line(line: str) -> bool:
    # Must have dictionary structure
    if "*" not in line:
        return False
    # Prefer at least 2 fields
    if line.count("*") >= 2:
        return True
    # Or contains marker tokens
    tokens = set(re.split(r"[\s\*]+", line))
    return bool(tokens.intersection(POS_TOKENS | CLASS_TOKENS | REGION_TOKENS))

def parse_multiword_headword(line: str, last_good_headword: Optional[str]) -> Optional[str]:
    """
    Extract headword from the 'headword zone' (before first '*').
    Allow 1-3 beja tokens (for cases like 'aada daatiya', 'aafimaab baakaay').
    If the zone starts with a non-beja token but last_good_headword exists and the
    first token looks like a subentry tail, return 'last_good_headword <tail>'.
    """
    if "*" not in line:
        return None

    head_zone = line.split("*", 1)[0].strip()
    if not head_zone:
        return None

    toks = [t for t in re.split(r"\s+", head_zone) if t]

    # Hard reject certain starts
    if toks and toks[0] in NEVER_HEADWORDS:
        return None
    if toks and toks[0] in GLOSS_START_STOPLIST:
        return None

    beja_toks = []
    for t in toks:
        t_clean = t.replace("\\", "")
        if BEJA_TOKEN_RE.match(t_clean) and t_clean not in GLOSS_START_STOPLIST and t_clean not in NEVER_HEADWORDS:
            beja_toks.append(t_clean)
        else:
            break

    if beja_toks:
        # allow up to 3 tokens in headword
        return " ".join(beja_toks[:3])

    # Repair case: head zone begins with something like 'daatiya' but missing its prefix
    # Only do this if we have a last_good_headword and the first token is beja-ish
    if last_good_headword and toks:
        t0 = toks[0].replace("\\", "")
        if BEJA_TOKEN_RE.match(t0) and t0 not in GLOSS_START_STOPLIST and t0 not in NEVER_HEADWORDS:
            return f"{last_good_headword} {t0}"

    return None

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
    last_good_headword = None

    def flush(end_line=None):
        nonlocal current, current_regions, blocks_written
        if current is None:
            return
        if end_line is not None:
            current["end_line"] = end_line

        # attach deduped regions
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

    for path, fname in tqdm(list(iter_page_files()), desc="Region-aware blocking v3"):
        page_num = int(fname.split("_")[1].split(".")[0])

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        lines = [normalize_line(l) for l in raw_lines]
        lines = [l for l in lines if not should_ignore_line(l)]

        for ln_idx, raw in enumerate(lines, start=1):
            line = raw

            # Treat sg/pl etc as continuation
            first_tok = line.split()[0] if line.split() else ""
            if first_tok in NEVER_HEADWORDS:
                if current is not None:
                    current["lines"].append(line)
                    current_regions.extend(extract_regions_from_line(line))
                    if is_region_line(line):
                        flush(end_line=ln_idx)
                continue

            # Strip leading digits if present (OCR junk)
            if line and line[0].isdigit():
                line = strip_leading_digits(line)

            # If region-only line ends entry
            if current is not None and is_region_line(line):
                current["lines"].append(line)
                current_regions.extend(extract_regions_from_line(line))
                flush(end_line=ln_idx)
                continue

            # Start of a new entry?
            if looks_like_entry_head_line(line):
                hw = parse_multiword_headword(line, last_good_headword)
                if hw:
                    # flush previous before starting
                    if current is not None:
                        flush(end_line=ln_idx - 1)

                    current = {
                        "page": page_num,
                        "start_line": ln_idx,
                        "headword_guess": hw,
                        "lines": [line],
                    }
                    current_regions.extend(extract_regions_from_line(line))
                    last_good_headword = hw

                    # If this line already contains region markers, entry is complete
                    if extract_regions_from_line(line):
                        flush(end_line=ln_idx)
                    continue

            # Otherwise continuation
            if current is not None:
                current["lines"].append(line)
                current_regions.extend(extract_regions_from_line(line))

                # If a continuation line contains regions AND has '*' structure, flush immediately
                if "*" in line and extract_regions_from_line(line):
                    flush(end_line=ln_idx)

        # Page boundary: flush anything still open at end of page
        if current is not None:
            flush(end_line=0)

    print(f"Wrote {blocks_written} blocks to {OUT_PATH}")

if __name__ == "__main__":
    main()
