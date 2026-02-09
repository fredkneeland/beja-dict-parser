# 4th try this time hopefully fixing the frist entry of the page bug
import os
import json
import re
from tqdm import tqdm
from typing import Optional

PAGES_DIR = "data/intermediate/pages_text_ocr_reflow"
OUT_PATH = "data/intermediate/blocks_jsonl/entry_blocks_region4.jsonl"

REGION_TOKENS = {"Er", "Su", "Eg", "Ar."}
CLASS_TOKENS = {"Sem", "Cush"}
POS_TOKENS = {"N", "V", "Adj", "Adv", "Pron", "Phr", "Intj"}

NEVER_HEADWORDS = {"sg", "pl", "pl.", "m", "f", "mf", "m.", "f.", "-", "—", "_"}
GLOSS_START_STOPLIST = {
    "woman", "women", "man", "men",
    "the", "a", "an", "of", "and", "or", "to", "from", "with", "in", "on", "at",
    "maturity",  # keep if it ever appears as line-start gloss
}

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
LEADING_DIGITS_RE = re.compile(r"^\s*\d+\s*(.*)$")
BEJA_TOKEN_RE = re.compile(r"^[a-z][a-z']*(?:/[a-z])?$")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")

def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("‎", "")
    line = line.strip()
    line = " ".join(line.split())
    return line

def fix_leading_star_noise(line: str) -> str:
    """
    Fix OCR artifacts where a line begins with something like:
      '*m_aagil * ...'  or '*f_word * ...'
    caused by column bleed. Convert to:
      'aagil * ...' or 'word * ...'
    Only applies when the line starts with '*<gender>_' and then a beja-ish token.
    """
    # Example: *m_aagil * ...
    m = re.match(r"^\*(?:m|f|n|mf)[ _]+([a-z][a-z']*(?:/[a-z])?)\b(.*)$", line)
    if m:
        hw = m.group(1)
        rest = m.group(2).lstrip()
        # If rest already begins with '*', keep it; otherwise add a space
        if rest.startswith("*"):
            return f"{hw} {rest}".strip()
        return f"{hw} {rest}".strip()

    # Example: *m aagil * ...
    m2 = re.match(r"^\*(?:m|f|n|mf)\s+([a-z][a-z']*(?:/[a-z])?)\b(.*)$", line)
    if m2:
        hw = m2.group(1)
        rest = m2.group(2).lstrip()
        if rest.startswith("*"):
            return f"{hw} {rest}".strip()
        return f"{hw} {rest}".strip()

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

def looks_like_entry_head_line_strong(line: str) -> bool:
    """Strong signal: >=2 stars or POS/class/region markers."""
    if "*" not in line:
        return False
    if line.count("*") >= 2:
        return True
    tokens = set(re.split(r"[\s\*]+", line))
    return bool(tokens.intersection(POS_TOKENS | CLASS_TOKENS | REGION_TOKENS))

def parse_multiword_headword(line: str, last_good_headword: Optional[str]) -> Optional[str]:
    if "*" not in line:
        return None

    head_zone = line.split("*", 1)[0].strip()
    if not head_zone:
        return None

    toks = [t for t in re.split(r"\s+", head_zone) if t]
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
        return " ".join(beja_toks[:3])

    # Repair: missing prefix like "daatiya * ..." after digit stripping
    if last_good_headword and toks:
        t0 = toks[0].replace("\\", "")
        if BEJA_TOKEN_RE.match(t0) and t0 not in GLOSS_START_STOPLIST and t0 not in NEVER_HEADWORDS:
            return f"{last_good_headword} {t0}"

    return None

def looks_like_entry_head_line_weak(line: str, hw: str) -> bool:
    """
    NEW: Accept entry-start lines that only have one '*' and no markers,
    e.g. 'aagreeb (m'atee...) * maturity,'.
    Criteria:
    - has '*'
    - headword parsed successfully
    - the text after '*' contains Latin or Arabic (gloss-like)
    """
    if "*" not in line:
        return False
    if hw in GLOSS_START_STOPLIST or hw in NEVER_HEADWORDS:
        return False

    after = line.split("*", 1)[1]
    if ARABIC_RE.search(after) or LATIN_RE.search(after):
        return True
    return False

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

        regs, seen = [], set()
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

    for path, fname in tqdm(list(iter_page_files()), desc="Region-aware blocking v4"):
        page_num = int(fname.split("_")[1].split(".")[0])

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        lines = [normalize_line(l) for l in raw_lines]
        lines = [l for l in lines if not should_ignore_line(l)]

        for ln_idx, raw in enumerate(lines, start=1):
            line = raw

            # Never allow sg/pl/etc to start an entry
            first_tok = line.split()[0] if line.split() else ""
            if first_tok in NEVER_HEADWORDS:
                if current is not None:
                    current["lines"].append(line)
                    current_regions.extend(extract_regions_from_line(line))
                    if is_region_line(line):
                        flush(end_line=ln_idx)
                continue

            # Strip digit prefixes (OCR junk)
            if line and line[0].isdigit():
                line = strip_leading_digits(line)

            # Fix leading '*m_' / '*f_' artifacts
            line = fix_leading_star_noise(line)

            # Region-only ends entry
            if current is not None and is_region_line(line):
                current["lines"].append(line)
                current_regions.extend(extract_regions_from_line(line))
                flush(end_line=ln_idx)
                continue

            # Try to start a new entry (strong OR weak)
            if "*" in line:
                hw = parse_multiword_headword(line, last_good_headword)
                if hw and (looks_like_entry_head_line_strong(line) or looks_like_entry_head_line_weak(line, hw)):
                    if current is not None:
                        flush(end_line=ln_idx - 1)

                    current = {"page": page_num, "start_line": ln_idx, "headword_guess": hw, "lines": [line]}
                    current_regions.extend(extract_regions_from_line(line))
                    last_good_headword = hw

                    # If this line contains regions, end immediately
                    if extract_regions_from_line(line):
                        flush(end_line=ln_idx)
                    continue

            # Continuation
            if current is not None:
                current["lines"].append(line)
                current_regions.extend(extract_regions_from_line(line))

                # If continuation has regions + '*' structure, treat as end marker
                if "*" in line and extract_regions_from_line(line):
                    flush(end_line=ln_idx)

        # Page boundary: flush anything still open at end of page
        if current is not None:
            flush(end_line=0)

    print(f"Wrote {blocks_written} blocks to {OUT_PATH}")

if __name__ == "__main__":
    main()