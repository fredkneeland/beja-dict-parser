import json
import os
import re
from typing import Optional, List, Dict, Any
from wordfreq import zipf_frequency

INPUT_BLOCKS = "data/intermediate/beja-en/blocks_jsonl/entry_blocks_region.jsonl"
OUT_JSONL = "data/output/dictionary.jsonl"
OUT_JSON = "data/output/dictionary.json"
OUT_ISSUES = "data/output/dictionary_issues.jsonl"

os.makedirs("data/output", exist_ok=True)

# Your abbreviation system
POS_TOKENS = {"Adj", "Adv", "Con", "Dem", "Intj", "N", "Num", "Phr", "Pps", "Pron", "V"}
REGION_TOKENS = {"Er", "Su", "Eg"}
CLASS_TOKENS = {"Cush", "Sem"}
GENDER_TOKENS = {"m", "f", "mf"}
NUMBER_TOKENS = {"sg", "pl", "pl."}

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
BEJA_TOKEN_RE = re.compile(r"^[a-z][a-z']*(?:/[a-z])?$")

def norm(s: str) -> str:
    # remove bidi/zero-width junk, collapse spaces
    s = s.replace("‎", "").replace("\u200f", "").replace("\u200e", "")
    s = s.replace("\u00a0", " ")
    s = " ".join(s.split())
    return s.strip()

def tokenize_star_text(lines: List[str]) -> List[str]:
    text = norm(" ".join(lines))
    parts = [p.strip() for p in text.split("*")]
    return [p for p in parts if p]

def extract_arabic_chunks(lines: List[str]) -> List[str]:
    text = norm(" ".join(lines))
    chunks = re.findall(r"[\u0600-\u06FF][\u0600-\u06FF\s،؛\(\)\-]*", text)
    out = []
    seen = set()
    for c in chunks:
        c = norm(c)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def normalize_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        t = norm(t)
        # strip punctuation that sticks to tags: "Er‏", "Su‏", etc.
        t = t.strip(".,;:!?'\"()[]{}<>")
        if t:
            out.append(t)
    return out

def english_zipf(tok: str) -> float:
    tok = tok.lower().strip(".,;:!?\"()[]{}")
    if not tok:
        return 0.0
    return zipf_frequency(tok, "en")

def looks_probably_english_headword(hw: str) -> bool:
    """
    Robust: if it's common English (Zipf >= ~4.5) and not very Beja-looking.
    For multiword headwords, if most tokens are common English, reject.
    """
    toks = hw.split()
    if not toks:
        return False

    # Beja-ish signals
    beja_signals = 0
    for t in toks:
        tl = t.lower()
        if "aa" in tl or "ii" in tl or "uu" in tl or "ee" in tl or "oo" in tl:
            beja_signals += 1
        if "'" in tl or tl.endswith("/t"):
            beja_signals += 1

    eng_scores = [english_zipf(t) for t in toks]
    if max(eng_scores) >= 5.0 and beja_signals == 0:
        return True
    # If all tokens are fairly common English and weak Beja signals
    if sum(1 for s in eng_scores if s >= 4.5) >= len(toks) and beja_signals == 0:
        return True
    return False

def parse_tags(tokens: List[str]) -> Dict[str, Any]:
    pos = []
    regions = []
    clazz = None
    gender = None
    number = None
    nominalized_verb = False

    for t in tokens:
        if t in POS_TOKENS:
            pos.append(t)
        elif t in REGION_TOKENS:
            regions.append(t)
        elif t in CLASS_TOKENS:
            clazz = t
        elif t in GENDER_TOKENS:
            gender = t
        elif t in NUMBER_TOKENS:
            number = t
        elif t == "N" and "V" in tokens:
            nominalized_verb = True

    # de-dupe
    def dedupe(xs):
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "pos": dedupe(pos) or None,
        "regions": dedupe(regions) or None,
        "class": clazz,
        "gender": gender,
        "number": number,
        "nominalized_verb": nominalized_verb,
    }

def extract_english_gloss(fields: List[str]) -> List[str]:
    """
    Keep only non-tag latin-ish gloss chunks, drop things like "Er Su Eg", "- Er", "= V".
    """
    glosses = []
    for f in fields:
        f = norm(f)
        if not f:
            continue

        # drop pure-tag fields
        toks = normalize_tokens(re.split(r"\s+", f))
        if toks and all(t in (POS_TOKENS | REGION_TOKENS | CLASS_TOKENS | GENDER_TOKENS | NUMBER_TOKENS | {"-", "—", "_", "=", "Ar."})
                        for t in toks):
            continue

        # If it contains latin letters but not Arabic, treat as English gloss candidate
        if re.search(r"[A-Za-z]", f) and not ARABIC_RE.search(f):
            # also drop cases like "- Er" or "= V"
            if len(toks) <= 2 and any(t in REGION_TOKENS or t in POS_TOKENS for t in toks):
                continue
            glosses.append(f)

    # de-dupe
    out, seen = [], set()
    for g in glosses:
        g = norm(g).strip(",;")
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out

def repair_missing_prefix(entry: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> bool:
    """
    Repair pattern like:
      prev.headword = "aada"
      entry.headword = "daatiya"   (should be "aada daatiya")
    Only apply when:
      - entry headword is single token, beja-ish
      - entry pos is Phr (or similar subentry-like)
      - previous exists, previous headword is single token
      - same regions (or entry regions missing but prev has them)
      - consecutive in source lines
    """
    if not prev:
        return False
    hw = entry.get("headword") or ""
    if " " in hw:
        return False
    if not BEJA_TOKEN_RE.match(hw):
        return False
    if entry.get("pos") != ["Phr"] and entry.get("pos") != ["Phr",]:
        # also allow None? but keep conservative
        return False

    prev_hw = prev.get("headword") or ""
    if " " in prev_hw:
        return False
    if not prev_hw:
        return False

    # line adjacency
    try:
        if entry["source"]["page"] != prev["source"]["page"]:
            return False
        if entry["source"]["start_line"] != prev["source"]["end_line"] + 1:
            return False
    except Exception:
        return False

    # region compatibility
    er = entry.get("regions")
    pr = prev.get("regions")
    if er and pr and er != pr:
        return False

    # apply
    entry["headword"] = f"{prev_hw} {hw}"
    entry["headword_parts"] = [prev_hw, hw]
    return True

def parse_block_to_entry(block: Dict[str, Any]) -> Dict[str, Any]:
    raw_lines = [norm(l) for l in block["lines"] if norm(l)]
    fields = tokenize_star_text(raw_lines)
    tokens = normalize_tokens(re.split(r"\s+", " ".join(fields)))

    tags = parse_tags(tokens)

    entry = {
        "headword": norm(block.get("headword_guess") or ""),
        "headword_parts": (block.get("headword_guess") or "").split() if block.get("headword_guess") else None,
        "gloss_en": None,
        "gloss_ar": None,
        "pos": tags["pos"],
        "gender": tags["gender"],
        "number": tags["number"],
        "class": tags["class"],
        "nominalized_verb": tags["nominalized_verb"],
        "regions": tags["regions"] or (block.get("regions_guess") or None),
        "raw": raw_lines,
        "source": {"page": block.get("page"), "start_line": block.get("start_line"), "end_line": block.get("end_line")},
    }

    # Arabic gloss
    ar = extract_arabic_chunks(raw_lines)
    entry["gloss_ar"] = ar or None

    # English gloss from fields AFTER headword-ish field
    # (fields[0] often includes headword + maybe junk; still safer to scan fields[1:])
    en = extract_english_gloss(fields[1:] if len(fields) > 1 else [])
    entry["gloss_en"] = en or None

    return entry

def main():
    entries: List[Dict[str, Any]] = []
    issues_out = open(OUT_ISSUES, "w", encoding="utf-8")

    prev_entry: Optional[Dict[str, Any]] = None

    with open(INPUT_BLOCKS, "r", encoding="utf-8") as f:
        for line in f:
            block = json.loads(line)
            entry = parse_block_to_entry(block)

            # If headword is missing or looks like English, treat it as continuation of previous
            hw = entry["headword"]
            if not hw or looks_probably_english_headword(hw):
                if prev_entry is not None:
                    prev_entry["raw"].extend(entry["raw"])
                    # Recompute glosses from merged raw (simple refresh)
                    merged = parse_block_to_entry({
                        "headword_guess": prev_entry["headword"],
                        "lines": prev_entry["raw"],
                        "page": prev_entry["source"]["page"],
                        "start_line": prev_entry["source"]["start_line"],
                        "end_line": prev_entry["source"]["end_line"],
                        "regions_guess": prev_entry.get("regions"),
                    })
                    # Keep the same headword, update glosses/tags if improved
                    for k in ["gloss_en", "gloss_ar", "pos", "gender", "number", "class", "regions", "nominalized_verb"]:
                        if merged.get(k) is not None:
                            prev_entry[k] = merged[k]

                    issues_out.write(json.dumps({
                        "type": "merged_continuation",
                        "dropped_headword": hw,
                        "into_headword": prev_entry["headword"],
                        "source": entry["source"],
                    }, ensure_ascii=False) + "\n")
                else:
                    issues_out.write(json.dumps({
                        "type": "dropped_orphan_block",
                        "reason": "english_or_missing_headword_and_no_previous",
                        "block": block,
                    }, ensure_ascii=False) + "\n")
                continue

            # Repair missing-prefix subentry (daatiya -> aada daatiya)
            if repair_missing_prefix(entry, prev_entry):
                issues_out.write(json.dumps({
                    "type": "repaired_subentry_prefix",
                    "new_headword": entry["headword"],
                    "source": entry["source"],
                }, ensure_ascii=False) + "\n")

            entries.append(entry)
            prev_entry = entry

    issues_out.close()

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for e in entries:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(OUT_JSON, "w", encoding="utf-8") as out:
        json.dump(entries, out, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries:")
    print(f" - {OUT_JSONL}")
    print(f" - {OUT_JSON}")
    print(f"Issues:")
    print(f" - {OUT_ISSUES}")

if __name__ == "__main__":
    main()