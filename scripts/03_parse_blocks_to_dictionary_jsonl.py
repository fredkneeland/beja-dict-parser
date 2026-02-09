# first try at making our dictionary blocks into JSONL dictionary entries
import json
import os
import re
from typing import List, Dict, Optional

INPUT_BLOCKS = "data/intermediate/blocks_jsonl/entry_blocks_region5.jsonl"
OUT_JSONL = "data/output/dictionary.jsonl"
OUT_JSON = "data/output/dictionary.json"
OUT_ERRORS = "data/output/dictionary_errors.jsonl"

os.makedirs("data/output", exist_ok=True)

REGION_TOKENS = {"Er", "Su", "Eg"}
CLASS_TOKENS = {"Cush", "Sem"}
POS_TOKENS = {"Adj", "Adv", "Con", "Dem", "Intj", "N", "Num", "Phr", "Pps", "Pron", "V"}

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")

def clean_token(t: str) -> str:
    return t.strip().strip(",;:")

def dedupe_preserve(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def split_star_fields(lines: List[str]) -> List[str]:
    # Join lines with spaces to avoid missing * boundaries across line breaks
    text = " ".join(lines)
    text = " ".join(text.split())
    parts = [p.strip() for p in text.split("*")]
    parts = [p for p in parts if p != ""]
    return parts

def extract_arabic_chunks(s: str) -> List[str]:
    # Grab contiguous Arabic runs; simple and OCR-tolerant
    chunks = re.findall(r"[\u0600-\u06FF][\u0600-\u06FF\s،؛\(\)\-]*", s)
    return dedupe_preserve([clean_token(c.replace("  ", " ").strip()) for c in chunks])

def extract_regions(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t in REGION_TOKENS]

def extract_class(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        if t in CLASS_TOKENS:
            return t
    return None

def extract_pos(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t in POS_TOKENS]

def extract_gender(tokens: List[str]) -> Optional[str]:
    # You see patterns like "m", "f", "mf" in blocks sometimes
    for t in tokens:
        if t in {"m", "f", "mf"}:
            return t
    return None

def parse_entry(block: Dict) -> (Optional[Dict], Optional[Dict]):
    lines = block["lines"]
    headword = block.get("headword_guess")

    # Star fields
    fields = split_star_fields(lines)

    # Tokenize all fields for tag extraction
    tokens = []
    for f in fields:
        tokens.extend(re.split(r"\s+", f))
    tokens = [clean_token(t) for t in tokens if t]

    regions = dedupe_preserve(block.get("regions_guess") or extract_regions(tokens))
    pos = dedupe_preserve(extract_pos(tokens))
    clazz = extract_class(tokens)
    gender = extract_gender(tokens)

    # Heuristic extraction of glosses:
    # Field order often: headword, english gloss, arabic gloss, POS, (gender), (class), regions
    # But OCR can scramble; we’ll do content-based detection:
    gloss_ar = []
    gloss_en = []

    # Arabic: scan full joined text
    gloss_ar = extract_arabic_chunks(" ".join(lines))

    # English gloss candidates: fields that contain latin letters but are not just tags
    for f in fields[1:]:  # skip headword-ish field
        if any(tag in f.split() for tag in POS_TOKENS | REGION_TOKENS | CLASS_TOKENS):
            # may still contain gloss, but often tag-only; continue scanning anyway
            pass
        if LATIN_RE.search(f) and not ARABIC_RE.search(f):
            # exclude pure tag fields
            if f in POS_TOKENS or f in REGION_TOKENS or f in CLASS_TOKENS:
                continue
            gloss_en.append(f)

    gloss_en = dedupe_preserve([clean_token(g) for g in gloss_en if g])

    entry = {
        "headword": headword,
        "headword_parts": headword.split() if headword else None,
        "gloss_en": gloss_en or None,
        "gloss_ar": gloss_ar or None,
        "pos": pos or None,
        "gender": gender,
        "class": clazz,
        "regions": regions or None,
        "raw": lines,
        "source": {
            "page": block.get("page"),
            "start_line": block.get("start_line"),
            "end_line": block.get("end_line"),
        },
    }

    # Validation: must have headword and at least something else useful
    if not entry["headword"]:
        return None, {"error": "missing_headword", "block": block}
    if not (entry["gloss_en"] or entry["gloss_ar"]):
        return entry, {"warning": "missing_gloss", "entry": entry}

    return entry, None

def main():
    entries = []
    with open(OUT_ERRORS, "w", encoding="utf-8") as err_out, \
         open(OUT_JSONL, "w", encoding="utf-8") as out:

        with open(INPUT_BLOCKS, "r", encoding="utf-8") as f:
            for line in f:
                block = json.loads(line)
                entry, issue = parse_entry(block)

                if issue:
                    err_out.write(json.dumps(issue, ensure_ascii=False) + "\n")

                if entry:
                    entries.append(entry)
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries to:")
    print(f" - {OUT_JSONL}")
    print(f" - {OUT_JSON}")
    print(f"Issues logged to:")
    print(f" - {OUT_ERRORS}")

if __name__ == "__main__":
    main()