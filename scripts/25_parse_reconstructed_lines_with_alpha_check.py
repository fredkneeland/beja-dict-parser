import json
import os
import re
from typing import Dict, Any, Optional, Tuple, List

IN_LINES = "data/intermediate/beja2/dict2_reconstructed_lines.jsonl"
OUT_JSONL = "data/output/dict2_clean.jsonl"
OUT_JSON = "data/output/dict2_clean.json"
OUT_ERRORS = "data/output/dict2_errors.jsonl"

os.makedirs("data/output", exist_ok=True)

LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# allow uppercase in OCR, but we normalize to lowercase
BEJA_HEADWORD_RE = re.compile(r"^[a-z][a-z'\/\-]{1,30}$")

# POS-ish markers that usually appear in real entries
POS_MARK_RE = re.compile(
    r"\b(n\.|v\.|adj\.|adv\.|v\.w\.|v\.s\.|v\.s5|ag\.|act\.|nm\.|nf\.|vm\.|vn\.)\b",
    re.IGNORECASE,
)

# Lines that are almost certainly not starting a new entry
CONTINUATION_PREFIX_RE = re.compile(r"^(def\.|cf|sg\.|pl\.|see|also)\b", re.IGNORECASE)

# Common junk “headwords” / labels
BAD_HEADWORDS = {
    "def", "def.", "sg", "pl", "adv", "adj", "n", "v", "cf", "eg", "er", "su",
    "m", "f", "mf", "act", "agent", "one", "the", "and", "or", "of", "to",
    "bias"
}

def norm(s: str) -> str:
    s = s.replace("‎", "").replace("\u200f", "").replace("\u200e", "")
    s = s.replace("\u202a", "").replace("\u202b", "").replace("\u202c", "")
    s = s.replace("\ufeff", "")
    return " ".join(s.split()).strip()

def normalize_headword(hw: str) -> str:
    hw = norm(hw).lower()
    hw = hw.strip(".,;:!?()[]{}<>\"“”")
    return hw

def normalize_key(s: str) -> str:
    s = norm(s).lower()
    s = re.sub(r"[^a-z']", "", s)
    return s

def first_initial(s: str) -> Optional[str]:
    k = normalize_key(s)
    for ch in k:
        if "a" <= ch <= "z":
            return ch
    return None

def next_letter(ch: str) -> Optional[str]:
    if not ch or len(ch) != 1:
        return None
    if ch == "z":
        return None
    return chr(ord(ch) + 1)

def extract_headword(raw_text: str) -> Optional[str]:
    t = norm(raw_text)
    if not t:
        return None
    first = t.split()[0]
    return first or None

def looks_like_real_entry_line(text: str) -> bool:
    """
    Heuristic: real entries almost always have a POS marker, or at least
    enough structure (commas / multiple tokens).
    """
    t = norm(text)
    if not t:
        return False

    # If line is just one token, it's usually not a full entry (often continuation/junk)
    if len(t.split()) <= 1:
        return False

    if POS_MARK_RE.search(t):
        return True

    # fallback: allow if it contains punctuation that suggests definition text
    if any(p in t for p in [",", ";", ":", "."]):
        return True

    return False

def headword_is_plausible(hw: str, raw_text: str) -> Tuple[bool, str]:
    if not hw:
        return False, "no_headword"

    text = norm(raw_text)

    # Reject obvious continuation lines
    if CONTINUATION_PREFIX_RE.match(text):
        return False, "continuation_line"

    # Must look like an entry line (prevents accepting "tusoosmooy" alone)
    if not looks_like_real_entry_line(text):
        return False, "not_entry_like_line"

    hw_norm = normalize_headword(hw)

    # reject if the headword token is basically a label/junk
    if hw_norm in BAD_HEADWORDS:
        return False, "headword_in_badlist"

    # reject if it contains digits or arabic
    if any(c.isdigit() for c in hw_norm) or ARABIC_RE.search(hw_norm):
        return False, "headword_has_digits_or_arabic"

    # must have latin letters
    if not LATIN_RE.search(hw_norm):
        return False, "headword_not_latin"

    # allow OCR uppercase by normalizing to lowercase, then enforce shape
    if not BEJA_HEADWORD_RE.match(hw_norm):
        return False, "headword_bad_shape"

    return True, ""

def parse_line_to_entry(rec: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = norm(rec.get("text", ""))
    if not text:
        return None, "empty_text"

    hw = extract_headword(text)
    ok, reason = headword_is_plausible(hw, text)
    if not ok:
        return None, reason

    hw_norm = normalize_headword(hw)

    m = POS_MARK_RE.search(text)
    pos_guess = m.group(1).lower() if m else None

    rest = text[len(text.split()[0]):].strip()

    entry = {
        "headword": hw_norm,  # normalize to lowercase for app/search
        "pos_guess": pos_guess,
        "gloss_en": rest if rest else None,
        "raw": text,
        "ocr_conf": rec.get("avg_conf", None),
        "source": {
            "page": rec.get("page"),
            "line_index": rec.get("line_index"),
            "psm": rec.get("psm"),
        },
    }
    return entry, None

def main():
    for p in (OUT_JSONL, OUT_ERRORS):
        if os.path.exists(p):
            os.remove(p)

    with open(IN_LINES, "r", encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    recs.sort(key=lambda r: (int(r.get("page", 0)), int(r.get("line_index", 0))))

    kept: List[Dict[str, Any]] = []
    last_good_entry: Optional[Dict[str, Any]] = None
    last_initial: Optional[str] = None

    consecutive_rejects = 0
    RESYNC_AFTER = 10

    def log_error(kind: str, rec: Dict[str, Any], entry: Optional[Dict[str, Any]] = None, note: str = ""):
        hw = entry["headword"] if entry and entry.get("headword") else (extract_headword(rec.get("text","")) or None)
        payload = {
            "headword": hw,
            "page": rec.get("page"),
            "line_index": rec.get("line_index"),
            "error": kind,
            "note": note,
            "avg_conf": rec.get("avg_conf"),
            "raw_text": rec.get("text"),
            "last_good_headword": last_good_entry["headword"] if last_good_entry else None,
        }
        with open(OUT_ERRORS, "a", encoding="utf-8") as out:
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

    for rec in recs:
        entry, err = parse_line_to_entry(rec)
        if err:
            consecutive_rejects += 1
            log_error(err, rec, None)
            continue

        cur_init = first_initial(entry["headword"])
        if cur_init is None:
            consecutive_rejects += 1
            log_error("no_initial", rec, entry)
            continue

        if last_initial is None:
            kept.append(entry)
            last_good_entry = entry
            last_initial = cur_init
            consecutive_rejects = 0
            continue

        # resync mode: after lots of rejects, accept next plausible and re-lock
        if consecutive_rejects >= RESYNC_AFTER:
            kept.append(entry)
            last_good_entry = entry
            last_initial = cur_init
            consecutive_rejects = 0
            continue

        allowed = {last_initial}
        nl = next_letter(last_initial)
        if nl:
            allowed.add(nl)

        if cur_init not in allowed:
            consecutive_rejects += 1
            log_error(
                "implausible_letter_jump",
                rec,
                entry,
                note=f"cur_initial='{cur_init}' not in allowed={sorted(list(allowed))}"
            )
            continue

        kept.append(entry)
        last_good_entry = entry
        last_initial = cur_init
        consecutive_rejects = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for e in kept:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(OUT_JSON, "w", encoding="utf-8") as out:
        json.dump(kept, out, ensure_ascii=False, indent=2)

    print(f"Kept {len(kept)} entries -> {OUT_JSONL} and {OUT_JSON}")
    print(f"Errors logged -> {OUT_ERRORS}")

if __name__ == "__main__":
    main()