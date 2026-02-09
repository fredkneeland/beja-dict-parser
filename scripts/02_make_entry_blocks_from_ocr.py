# first try with text copy results

import os
import json
import re
from tqdm import tqdm

PAGES_DIR = "data/intermediate/pages_text_ocr_reflow"
OUT_PATH = "data/intermediate/blocks_jsonl/entry_blocks.jsonl"

# More permissive headword token (Beja latin)
HEADWORD_AT_START = re.compile(r"^(?P<hw>[a-z][a-z'\\]+(?:/[a-z])?)\b")

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
DIGIT_START_RE = re.compile(r"^\s*\d")

# What makes a line "entry-like"
POS_TOKENS = {"N", "V", "Adj", "Adv", "Pron", "Phr", "Intj"}
REGION_TOKENS = {"Er", "Su", "Eg", "Ar."}
CLASS_TOKENS = {"Sem", "Cush"}

# English words that commonly appear as glosses at line-start when OCR scrambles (add over time)
GLOSS_START_STOPLIST = {
    "woman", "women", "man", "men", "boy", "girl",
    "the", "a", "an", "of", "and", "or",
    "in", "on", "at", "to", "from", "with",
    "maturity", "health", "custom", "culture", "only",
}

def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("‎", "")  # bidi mark
    line = line.strip()
    line = " ".join(line.split())
    return line

def normalize_headword(hw: str) -> str:
    hw = hw.replace("\\", "")
    hw = hw.replace(" ", "")
    return hw

def should_ignore_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_NUMBER_RE.match(line):
        return True
    return False

def looks_like_entry_head(line: str, hw: str) -> bool:
    """
    Decide whether this line is the *start* of a dictionary entry.
    We require the line to contain a '*' separator soon, and to contain
    POS/region/class markers somewhere (not necessarily same line, but this helps a lot).
    """
    # If the "headword" is actually a common English gloss word, reject
    if hw in GLOSS_START_STOPLIST:
        return False

    # Real entries are usually lowercase and not too English-looking.
    # Reject obvious English plural patterns (like women/men already caught, but keep this)
    if len(hw) <= 3 and hw in {"the", "and", "for", "with"}:
        return False

    # Strong requirement: must have '*' separator on the same line
    # (Your data has this very consistently at entry starts.)
    if "*" not in line:
        return False

    # Require at least 2 stars on the line (headword * gloss * ...).
    if line.count("*") < 2:
        return False

    # Look for POS/region/class tokens anywhere on the line
    tokens = set(re.split(r"\s+", line.replace("*", " ")))
    if tokens.intersection(POS_TOKENS | REGION_TOKENS | CLASS_TOKENS):
        return True

    # If no tags on the same line, still allow if it "looks like" a headword line:
    # e.g. headword * English gloss * Arabic gloss
    # We'll accept if there is Arabic on the line (good signal) OR if a POS appears very shortly after in later lines (handled by continuation).
    if re.search(r"[\u0600-\u06FF]", line):
        return True

    return False

def iter_page_files():
    files = [f for f in os.listdir(PAGES_DIR) if f.startswith("page_") and f.endswith(".txt")]
    files.sort()
    for f in files:
        yield os.path.join(PAGES_DIR, f), f

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    blocks_written = 0
    current = None

    # Clear output on rerun
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    for path, fname in tqdm(list(iter_page_files()), desc="Block pages"):
        page_num = int(fname.split("_")[1].split(".")[0])

        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        lines = [normalize_line(l) for l in raw_lines]
        lines = [l for l in lines if not should_ignore_line(l)]

        for ln_idx, line in enumerate(lines, start=1):
            # Digit-leading lines: treat as continuation if we have a block
            if DIGIT_START_RE.match(line):
                if current is not None and "*" in line:
                    current["lines"].append(line)
                continue

            m = HEADWORD_AT_START.match(line)
            if m:
                hw = normalize_headword(m.group("hw"))

                if looks_like_entry_head(line, hw):
                    # flush previous
                    if current is not None:
                        current["end_line"] = ln_idx - 1
                        with open(OUT_PATH, "a", encoding="utf-8") as out:
                            out.write(json.dumps(current, ensure_ascii=False) + "\n")
                        blocks_written += 1

                    current = {
                        "page": page_num,
                        "start_line": ln_idx,
                        "headword_guess": hw,
                        "lines": [line],
                    }
                else:
                    # Not a real entry head; treat as continuation text
                    if current is not None:
                        current["lines"].append(line)
            else:
                # continuation
                if current is not None:
                    current["lines"].append(line)

    # flush last
    if current is not None:
        with open(OUT_PATH, "a", encoding="utf-8") as out:
            out.write(json.dumps(current, ensure_ascii=False) + "\n")
        blocks_written += 1

    print(f"Wrote {blocks_written} entry blocks to {OUT_PATH}")

if __name__ == "__main__":
    main()