import json
import os
import re
from typing import List, Dict, Any, Optional

IN_BLOCKS = "data/intermediate/en-beja/english_beja_blocks.jsonl"
OUT_JSONL = "data/output/english_beja.jsonl"
OUT_JSON = "data/output/english_beja.json"
OUT_ISSUES = "data/output/english_beja_issues.jsonl"

os.makedirs("data/output", exist_ok=True)

REGIONS = {"Er", "Su", "Eg"}
POS = {"Adj", "Adv", "Con", "Dem", "Intj", "N", "Num", "Phr", "Pps", "Pron", "V"}
CLASSES = {"Cush", "Sem"}
GENDER = {"m", "f", "mf"}
NUMBER = {"sg", "pl", "pl."}

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")

def norm(s: str) -> str:
    s = s.replace("‎", "").replace("\u200f", "").replace("\u200e", "")
    s = s.replace("\u202a", "").replace("\u202b", "").replace("\u202c", "")
    s = s.replace("\ufeff", "")
    s = " ".join(s.split()).strip()
    return s

def split_star_fields(lines: List[str]) -> List[str]:
    text = norm(" ".join(lines))
    return [f.strip() for f in text.split("*") if f.strip()]

def dedupe(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_arabic(lines: List[str]) -> List[str]:
    text = norm(" ".join(lines))
    chunks = re.findall(r"[\u0600-\u06FF][\u0600-\u06FF\s،؛\(\)\-]*", text)
    out, seen = [], set()
    for c in chunks:
        c = norm(c)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def tokenize(fields: List[str]) -> List[str]:
    toks: List[str] = []
    for f in fields:
        toks.extend(norm(f).split())
    # normalize tokens like "Er‏"
    cleaned = []
    for t in toks:
        t = t.strip(".,;:!?\"()[]{}<>")
        if t:
            cleaned.append(t)
    return cleaned

def normalize_regions(tokens: List[str]) -> List[str]:
    # Fix common OCR: "S" used for "Su" in region lists
    # e.g. "... Er S Eg" -> treat S as Su if surrounded by regions
    out = []
    for t in tokens:
        if t == "S":
            out.append("Su")
        else:
            out.append(t)
    regs = [t for t in out if t in REGIONS]
    return dedupe(regs)

def normalize_gender(tokens: List[str]) -> Optional[str]:
    # Handle "mf?" or "m?" etc.
    for t in tokens:
        t2 = t.replace("?", "")
        if t2 in GENDER:
            return t2
    return None

def normalize_class(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        if t in CLASSES:
            return t
    return None

def normalize_pos(tokens: List[str]) -> List[str]:
    # Fix OCR: "Ady" -> "Adv"
    out = []
    for t in tokens:
        if t == "Ady":
            out.append("Adv")
        elif t in POS:
            out.append(t)
    return dedupe(out)

def split_beja_forms(s: str) -> List[str]:
    # Common separators: comma, semicolon
    s = norm(s)
    parts = re.split(r"[;,]\s*", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def parse_block(block: Dict[str, Any]) -> (Optional[Dict[str, Any]], Optional[Dict[str, Any]]):
    raw_lines = [norm(l) for l in block.get("lines", []) if norm(l)]
    if not raw_lines:
        return None, {"type": "empty_block", "block": block}

    fields = split_star_fields(raw_lines)
    if len(fields) < 2:
        return None, {"type": "too_few_fields", "raw": raw_lines, "source": block}

    # Heuristic for English→Beja layout:
    # english * beja * (arabic...) * (tags...) * regions
    english = fields[0]
    beja_field = fields[1]

    toks = tokenize(fields)
    regions = normalize_regions(toks)
    pos = normalize_pos(toks)
    clazz = normalize_class(toks)
    gender = normalize_gender(toks)

    arabic = extract_arabic(raw_lines)

    entry = {
        "english": english,
        "beja": split_beja_forms(beja_field),
        "gloss_ar": arabic or None,
        "pos": pos or None,
        "gender": gender,
        "class": clazz,
        "regions": regions or None,
        "raw": raw_lines,
        "source": {
            "page": block.get("page"),
            "start_line": block.get("start_line"),
            "end_line": block.get("end_line"),
        },
    }

    # Basic validation
    if not entry["english"] or not entry["beja"]:
        return None, {"type": "missing_core_fields", "entry": entry}

    return entry, None

def main():
    entries: List[Dict[str, Any]] = []
    issues_out = open(OUT_ISSUES, "w", encoding="utf-8")

    with open(IN_BLOCKS, "r", encoding="utf-8") as f:
        for line in f:
            block = json.loads(line)
            entry, issue = parse_block(block)
            if issue:
                issues_out.write(json.dumps(issue, ensure_ascii=False) + "\n")
            if entry:
                entries.append(entry)

    issues_out.close()

    # Write JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for e in entries:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Write single JSON array (bundle-friendly)
    with open(OUT_JSON, "w", encoding="utf-8") as out:
        json.dump(entries, out, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries:")
    print(f" - {OUT_JSONL}")
    print(f" - {OUT_JSON}")
    print(f"Issues logged to:")
    print(f" - {OUT_ISSUES}")

if __name__ == "__main__":
    main()