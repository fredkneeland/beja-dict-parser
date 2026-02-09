#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional fast fuzzy matching
try:
    from rapidfuzz import fuzz, process  # type: ignore
    HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    HAS_RAPIDFUZZ = False


@dataclass
class Entry:
    headword: str
    gloss_en: List[str]
    gloss_ar: List[str]
    pos: List[str]
    regions: List[str]
    page: Optional[int]
    start_line: Optional[int]
    end_line: Optional[int]


def normalize_text(s: str) -> str:
    """
    Normalize for matching:
    - lowercase
    - collapse whitespace
    - keep apostrophes and /t (important for your headwords)
    - strip most punctuation
    """
    s = s.replace("‎", "").replace("\u200f", "").replace("\u200e", "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Keep letters, apostrophe, slash, spaces
    s = re.sub(r"[^a-z'\/\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_entries(jsonl_path: str) -> List[Entry]:
    entries: List[Entry] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            headword = (obj.get("headword") or "").strip()
            if not headword:
                continue

            src = obj.get("source") or {}
            entries.append(
                Entry(
                    headword=headword,
                    gloss_en=(obj.get("gloss_en") or []) if isinstance(obj.get("gloss_en"), list) else [],
                    gloss_ar=(obj.get("gloss_ar") or []) if isinstance(obj.get("gloss_ar"), list) else [],
                    pos=(obj.get("pos") or []) if isinstance(obj.get("pos"), list) else [],
                    regions=(obj.get("regions") or []) if isinstance(obj.get("regions"), list) else [],
                    page=src.get("page"),
                    start_line=src.get("start_line"),
                    end_line=src.get("end_line"),
                )
            )
    return entries


def format_entry(e: Entry, score: Optional[float] = None) -> str:
    score_txt = f"[score {score:.1f}] " if score is not None else ""
    en = "; ".join(e.gloss_en) if e.gloss_en else ""
    ar = " | ".join(e.gloss_ar) if e.gloss_ar else ""
    pos = ",".join(e.pos) if e.pos else ""
    regs = ",".join(e.regions) if e.regions else ""
    loc = f"p{e.page}:{e.start_line}-{e.end_line}" if e.page is not None else "p?:?-?"
    bits = [
        f"{score_txt}{e.headword}",
        f"  EN: {en}" if en else "  EN: (none)",
        f"  AR: {ar}" if ar else "  AR: (none)",
        f"  TAGS: pos={pos or '-'} regions={regs or '-'}",
        f"  SOURCE: {loc}",
    ]
    return "\n".join(bits)


def best_matches(
    query: str,
    entries: List[Entry],
    top_k: int = 10,
    min_score: float = 60.0,
    mode: str = "headword",
) -> List[Tuple[float, Entry]]:
    """
    mode:
      - headword: fuzzy match only on headword
      - all: fuzzy match on headword + english + arabic (slower, but helpful)
    """
    qn = normalize_text(query)

    if not qn:
        return []

    # Build search strings
    def entry_search_text(e: Entry) -> str:
        if mode == "headword":
            return normalize_text(e.headword)
        # include gloss_en + gloss_ar for broader search
        en = " ".join(e.gloss_en or [])
        ar = " ".join(e.gloss_ar or [])
        # Arabic won’t normalize the same way; keep it raw appended too
        return normalize_text(e.headword + " " + en) + " " + ar

    search_space = [(entry_search_text(e), idx) for idx, e in enumerate(entries)]
    # Filter empty keys
    search_space = [(k, idx) for (k, idx) in search_space if k.strip()]

    results: List[Tuple[float, Entry]] = []

    if HAS_RAPIDFUZZ:
        keys = [k for (k, _) in search_space]
        # Use token_set_ratio for robustness with multiword headwords and glosses
        matches = process.extract(
            qn,
            keys,
            scorer=fuzz.token_set_ratio,
            limit=top_k * 5,  # grab more then filter
        )
        for key, score, key_index in matches:
            idx = search_space[key_index][1]
            if score >= min_score:
                results.append((float(score), entries[idx]))
            if len(results) >= top_k:
                break
    else:
        # Fallback using difflib (slower, weaker)
        scored: List[Tuple[float, Entry]] = []
        for k, idx in search_space:
            score = difflib.SequenceMatcher(a=qn, b=k).ratio() * 100.0
            if score >= min_score:
                scored.append((score, entries[idx]))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[:top_k]

    # De-dup identical headwords (keep best score)
    best: Dict[str, Tuple[float, Entry]] = {}
    for s, e in results:
        if e.headword not in best or s > best[e.headword][0]:
            best[e.headword] = (s, e)
    out = list(best.values())
    out.sort(key=lambda x: x[0], reverse=True)
    return out[:top_k]


def prefix_matches(query: str, entries: List[Entry], top_k: int = 10) -> List[Entry]:
    qn = normalize_text(query)
    if not qn:
        return []
    hits = []
    for e in entries:
        hn = normalize_text(e.headword)
        if hn.startswith(qn):
            hits.append(e)
            if len(hits) >= top_k:
                break
    return hits


def main():
    ap = argparse.ArgumentParser(description="Beja dictionary lookup (fuzzy + exact).")
    ap.add_argument("query", help="Search query, e.g. aab or 'aada daatiya'")
    ap.add_argument("--dict", default="data/output/dictionary.jsonl", help="Path to dictionary.jsonl")
    ap.add_argument("--top", type=int, default=10, help="Number of results to show")
    ap.add_argument("--min", type=float, default=60.0, help="Minimum fuzzy score (0-100)")
    ap.add_argument("--mode", choices=["headword", "all"], default="headword",
                    help="Match on headword only (fast) or headword+gloss (broader)")
    ap.add_argument("--prefix", action="store_true", help="Prefix search only (no fuzzy)")
    args = ap.parse_args()

    if not os.path.exists(args.dict):
        raise SystemExit(f"Dictionary file not found: {args.dict}")

    entries = load_entries(args.dict)

    if args.prefix:
        hits = prefix_matches(args.query, entries, top_k=args.top)
        if not hits:
            print("No prefix matches.")
            return
        for e in hits:
            print(format_entry(e))
            print("-" * 60)
        return

    matches = best_matches(args.query, entries, top_k=args.top, min_score=args.min, mode=args.mode)

    if not matches:
        print("No matches above threshold.")
        # As a helpful fallback, show a few prefix hits
        pref = prefix_matches(args.query, entries, top_k=min(args.top, 5))
        if pref:
            print("\nPrefix matches:")
            for e in pref:
                print(format_entry(e))
                print("-" * 60)
        return

    for score, e in matches:
        print(format_entry(e, score=score))
        print("-" * 60)


if __name__ == "__main__":
    main()