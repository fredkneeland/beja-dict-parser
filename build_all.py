#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import runpy
import time

def run_script(script_path: Path) -> float:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n▶ Running {script_path.name}")
    start = time.perf_counter()

    runpy.run_path(str(script_path), run_name="__main__")

    elapsed = time.perf_counter() - start
    print(f"✅ Finished {script_path.name} in {elapsed:.2f}s")

    return elapsed

def main() -> int:
    repo_root = Path(__file__).resolve().parents[0]
    scripts_dir = repo_root / "scripts"

    # ✅ Update these to match your filenames, in the order you want
    pipeline = [
        scripts_dir / "01_extract_pdf_text.py",
        scripts_dir / "02_make_entry_blocks_region_aware.py",
        scripts_dir / "03_parse_blocks_to_dictionary_jsonl.py",
        scripts_dir / "11_render_english_beja_pdf.py",
        scripts_dir / "12_ocr_english_beja_pages.py", 
        scripts_dir / "13_block_english_beja_by_region.py", 
        scripts_dir / "14_make_english_beja_dictionary_json.py", 
        scripts_dir / "21_render_pdf.py", 
        scripts_dir / "22_preprocess_pages.py", 
        scripts_dir / "23_ocr_pages_to_data.py", 
        scripts_dir / "24_reconstruct_lines_from_ocr.py", 
        scripts_dir / "25_parse_reconstructed_lines_with_alpha_check.py",
    ]

    print("Building all JSON artifacts...")
    try:
        for script in pipeline:
            run_script(script)
    except Exception as e:
        print(f"\n❌ Build failed: {e}", file=sys.stderr)
        return 1

    print("\n🎉 All builds completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())