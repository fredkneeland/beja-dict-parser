#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Set, Dict, List
from collections import Counter
import re

_TRAILING_PUNCT = ".,;:!?)\"]}»›"
_LEADING_PUNCT = "([\"{«‹"

def normalize_beja_word(w: str) -> str:
    """
    Normalize a Beja token for set comparison.

    Rules:
    - preserve apostrophes (meaningful in Beja)
    - lowercase
    - trim whitespace
    - remove a trailing '/t' only (e.g., 'kwidhaab/t' -> 'kwidhaab')
    - ignore common surrounding punctuation (but not internal apostrophes)
    """
    s = w.strip().lower()

    # strip surrounding punctuation that might come from parsing/OCR (safe)
    s = s.strip(_TRAILING_PUNCT).strip(_LEADING_PUNCT)

    # remove ONLY a final '/t'
    s = re.sub(r"/t$", "", s)

    return s


def extract_beja_words(obj: dict) -> List[str]:
    """
    Supports common schemas:
      - {"headword": "..."}
      - {"beja": ["...","..."]}
    Returns a list (possibly empty).
    """
    words: List[str] = []

    hw = obj.get("headword")
    if isinstance(hw, str) and hw.strip():
        words.append(hw)

    b = obj.get("beja")
    if isinstance(b, list):
        for item in b:
            if isinstance(item, str) and item.strip():
                words.append(item)
    elif isinstance(b, str) and b.strip():
        # sometimes "beja" can be a single string
        words.append(b)

    return words


def load_beja_set(jsonl_path: Path) -> tuple[Set[str], Counter]:
    """
    Returns:
      - set of unique normalized Beja words
      - Counter of raw normalized word frequencies (helps debug duplicates)
    """
    beja_set: Set[str] = set()
    freq = Counter()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{jsonl_path} line {line_no}: invalid JSON ({e})") from e

            for w in extract_beja_words(obj):
                nw = normalize_beja_word(w)
                if nw:
                    beja_set.add(nw)
                    freq[nw] += 1

    return beja_set, freq


def write_word_list(path: Path, words: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(w + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Beja headword coverage across three dictionary JSONL files."
    )
    parser.add_argument("dict1", type=Path, help="Path to dictionary 1 JSONL")
    parser.add_argument("dict2", type=Path, help="Path to dictionary 2 JSONL")
    parser.add_argument("dict3", type=Path, help="Path to dictionary 3 JSONL")
    parser.add_argument("--outdir", type=Path, default=Path("reports/dict_compare"),
                        help="Directory for output reports")
    args = parser.parse_args()

    for p in [args.dict1, args.dict2, args.dict3]:
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 1

    sets: Dict[str, Set[str]] = {}
    freqs: Dict[str, Counter] = {}

    paths = {"dict1": args.dict1, "dict2": args.dict2, "dict3": args.dict3}

    for name, path in paths.items():
        s, c = load_beja_set(path)
        sets[name] = s
        freqs[name] = c
        print(f"{name}: {len(s)} unique Beja words (from {path})")

    A, B, C = sets["dict1"], sets["dict2"], sets["dict3"]

    # For each dictionary, compute:
    # - in_both_others: appears in both other dictionaries
    # - in_any_other: appears in at least one other dictionary
    # - unique_to_this: appears only here
    stats = {}

    def compute_stats(this: Set[str], other1: Set[str], other2: Set[str]) -> dict:
        in_both = this & other1 & other2
        in_any = this & (other1 | other2)
        unique = this - (other1 | other2)
        return {
            "total_unique": len(this),
            "in_both_others": len(in_both),
            "in_any_other": len(in_any),
            "unique_to_this": len(unique),
        }

    stats["dict1"] = compute_stats(A, B, C)
    stats["dict2"] = compute_stats(B, A, C)
    stats["dict3"] = compute_stats(C, A, B)

    # Global overlap info (optional but handy)
    overlap = {
        "all_three": len(A & B & C),
        "dict1_and_dict2": len(A & B),
        "dict1_and_dict3": len(A & C),
        "dict2_and_dict3": len(B & C),
        "union_all": len(A | B | C),
    }

    # Unique word lists per dictionary
    unique1 = A - (B | C)
    unique2 = B - (A | C)
    unique3 = C - (A | B)

    outdir: Path = args.outdir
    write_word_list(outdir / "unique_dict1.txt", unique1)
    write_word_list(outdir / "unique_dict2.txt", unique2)
    write_word_list(outdir / "unique_dict3.txt", unique3)

    summary = {
        "inputs": {k: str(v) for k, v in paths.items()},
        "stats": stats,
        "overlap": overlap,
        "outputs": {
            "unique_dict1": str(outdir / "unique_dict1.txt"),
            "unique_dict2": str(outdir / "unique_dict2.txt"),
            "unique_dict3": str(outdir / "unique_dict3.txt"),
        },
        "notes": {
            "normalization": "strip + lowercase (edit normalize_beja_word() if you want more aggressive rules)"
        }
    }

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "dict_overlap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for k in ("dict1", "dict2", "dict3"):
        s = stats[k]
        print(
            f"{k}: total={s['total_unique']} | in_any_other={s['in_any_other']} | "
            f"in_both_others={s['in_both_others']} | unique={s['unique_to_this']}"
        )
    print(f"all_three overlap: {overlap['all_three']}")
    print(f"union_all: {overlap['union_all']}")
    print(f"\nWrote: {summary_path}")
    print(f"Wrote unique lists into: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())