# 5th try this time trying to get rid fo the English words
import os
import json
import re
from tqdm import tqdm
from typing import Optional
from wordfreq import zipf_frequency

PAGES_DIR = "data/intermediate/pages_text_ocr_reflow"
OUT_PATH = "data/intermediate/blocks_jsonl/entry_blocks_region5.jsonl"

# --- Abbreviations / tags from your dictionary ---
POS_TOKENS = {"Adj", "Adv", "Con", "Dem", "Intj", "N", "Num", "Phr", "Pps", "Pron", "V"}
REGION_TOKENS = {"Er", "Su", "Eg"}
CLASS_TOKENS = {"Cush", "Sem"}

# Continuation tokens that should never be treated as headwords
NEVER_HEADWORDS = {
    "sg", "pl", "pl.", "m", "f", "mf", "m.", "f.", "-", "—", "_", "=",
    # also: the abbreviation tokens should never be headwords
    *POS_TOKENS, *REGION_TOKENS, *CLASS_TOKENS
}

# English words that often appear as wrapped gloss lines
ENGLISH_STOPLIST = {
    "woman", "women", "man", "men", "boy", "girl",
    "the", "a", "an", "of", "and", "or", "to", "from", "with", "in", "on", "at",
    "most", "time", "anyway", "usually", "only",
    "maturity", "health", "custom", "culture",
    "at night", "something wrong", "other", "sips in turn",
}

BEJA_STRONG_PATTERNS = (
    re.compile(r"(aa|ii|uu|oo|ee)"),   # lots of Beja entries have long vowels doubled
    re.compile(r"/t$"),               # dictionary suffix like aabkaab/t
    re.compile(r"'"),                 # apostrophe appears in Beja romanization
)

def beja_strength_score(token: str) -> int:
    t = token.lower()
    score = 0
    for rx in BEJA_STRONG_PATTERNS:
        if rx.search(t):
            score += 1
    return score

def english_strength_score(token: str) -> float:
    """
    Uses wordfreq Zipf scale for English.
    Typical values:
      ~6-7 very common words (the, and, of...)
      ~4-5 common (woman, people, time...)
      ~1-3 rare
      0 unknown
    """
    t = token.lower().strip(".,;:!?\"()[]{}")
    if not t:
        return 0.0
    return zipf_frequency(t, "en")

def is_probably_english_token(token: str) -> bool:
    """
    Robust English test:
    - If token is a common English word (Zipf >= 4.5)
    - AND it doesn't have strong Beja orthography signals
    then treat as English.
    """
    t = token.lower()
    eng = english_strength_score(t)
    beja = beja_strength_score(t)

    # Strong English: very likely a gloss
    if eng >= 5.0 and beja <= 0:
        return True

    # Moderately common English: reject unless Beja-looking
    if eng >= 4.5 and beja <= 0:
        return True

    return False

def is_englishy_headword(hw: str) -> bool:
    """
    Headword may be 1–3 tokens.
    Rule:
      - If ALL tokens look English-ish (or at least one is strongly English)
      - and the headword has weak Beja signals overall
      => reject as headword
    """
    tokens = hw.split()
    if not tokens:
        return False

    eng_scores = [english_strength_score(t) for t in tokens]
    beja_scores = [beja_strength_score(t) for t in tokens]

    # print("English scores:", eng_scores, "Beja scores:", beja_scores, "for headword:", hw)

    # If any token is very common English and no token is strongly Beja, reject
    if max(eng_scores) >= 5.0 and max(beja_scores) <= 0:
        return True

    # If most tokens are common English and Beja signals are weak, reject
    common_eng = sum(1 for s in eng_scores if s >= 4.5)
    if common_eng >= max(1, len(tokens) - 0):  # basically "all tokens"
        if max(beja_scores) <= 0:
            return True

    return False

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
LEADING_DIGITS_RE = re.compile(r"^\s*\d+\s*(.*)$")

# Beja-ish token: lowercase latin with optional apostrophe or /t
BEJA_TOKEN_RE = re.compile(r"^[a-z][a-z']*(?:/[a-z])?$")

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")

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
    # region-only line like "Er Su Eg"
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

def fix_leading_star_noise(line: str) -> str:
    """
    Fix OCR artifacts like:
      '*m_aagil * ...'  -> 'aagil * ...'
      '*f_word * ...'   -> 'word * ...'
      '*m aagil * ...'  -> 'aagil * ...'
    """
    m = re.match(r"^\*(?:m|f|n|mf)[ _]+([a-z][a-z']*(?:/[a-z])?)\b(.*)$", line)
    if m:
        hw = m.group(1)
        rest = m.group(2).lstrip()
        return f"{hw} {rest}".strip()

    m2 = re.match(r"^\*(?:m|f|n|mf)\s+([a-z][a-z']*(?:/[a-z])?)\b(.*)$", line)
    if m2:
        hw = m2.group(1)
        rest = m2.group(2).lstrip()
        return f"{hw} {rest}".strip()

    return line

def looks_like_entry_head_line_strong(line: str) -> bool:
    """
    Strong signal: >=2 stars OR has POS/class/region tags on same line.
    """
    if "*" not in line:
        return False
    if line.count("*") >= 2:
        return True
    tokens = set(re.split(r"[\s\*]+", line))
    return bool(tokens.intersection(POS_TOKENS | CLASS_TOKENS | REGION_TOKENS))

def looks_like_entry_head_line_weak(line: str, hw: str) -> bool:
    """
    Weak signal: one-star lines like 'aagreeb ... * maturity,'
    Accept only if after '*' contains gloss-like text (Latin/Arabic).
    """
    if "*" not in line:
        return False
    if hw in NEVER_HEADWORDS:
        return False
    if is_englishy_headword(hw):
        print("Rejecting Englishy headword:", hw)
        return False
    after = line.split("*", 1)[1]
    return bool(ARABIC_RE.search(after) or LATIN_RE.search(after))

def parse_multiword_headword(line: str, last_good_headword: Optional[str]) -> Optional[str]:
    """
    Parse 1-3 beja tokens before first '*': allows 'aada daatiya', 'aafimaab baakaay'.
    Also allows repair: if head_zone begins with a beja-ish token but missing prefix, prepend last_good_headword.
    """
    if "*" not in line:
        return None

    head_zone = line.split("*", 1)[0].strip()
    if not head_zone:
        return None

    toks = [t for t in re.split(r"\s+", head_zone) if t]
    if not toks:
        return None

    # Never allow these as headword starts
    if toks[0] in NEVER_HEADWORDS:
        return None

    # Collect beja-ish tokens (up to 3)
    beja_toks = []
    for t in toks:
        t_clean = t.replace("\\", "")
        if BEJA_TOKEN_RE.match(t_clean) and t_clean not in NEVER_HEADWORDS:
            beja_toks.append(t_clean)
        else:
            break

    if beja_toks:
        hw = " ".join(beja_toks[:3])
        # Reject if it's clearly an English word and not beja-looking
        if is_englishy_headword(hw):
            print("2Rejecting Englishy headword:", hw)
            return None
        return hw

    # Repair: missing prefix on subentry tail, e.g. 'daatiya * ...'
    if last_good_headword and BEJA_TOKEN_RE.match(toks[0]):
        t0 = toks[0].replace("\\", "")
        if t0 not in NEVER_HEADWORDS and not is_englishy_headword(t0):
            return f"{last_good_headword} {t0}"
        else:
            print("3Rejecting Englishy headword repair:", t0)

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

    for path, fname in tqdm(list(iter_page_files()), desc="Region-aware blocking v5"):
        page_num = int(fname.split("_")[1].split(".")[0])

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        lines = [normalize_line(l) for l in raw_lines]
        lines = [l for l in lines if not should_ignore_line(l)]

        for ln_idx, raw in enumerate(lines, start=1):
            line = raw

            # Digit junk and leading '*m_' noise fixes
            if line and line[0].isdigit():
                line = strip_leading_digits(line)
            line = fix_leading_star_noise(line)

            # Continuation tokens should never start entries
            first_tok = line.split()[0] if line.split() else ""
            if first_tok in NEVER_HEADWORDS:
                if current is not None:
                    current["lines"].append(line)
                    current_regions.extend(extract_regions_from_line(line))
                    if is_region_line(line):
                        flush(end_line=ln_idx)
                continue

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