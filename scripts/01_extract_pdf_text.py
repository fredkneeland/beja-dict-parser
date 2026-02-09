import os
import json
import math
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple

INPUT_PDF = "data/input/beja-dictionary-simple.pdf"
OUT_DIR = "data/intermediate/pages_text_ocr_reflow"

DPI = 350
TESS_LANG = "ara+eng"

# Ask for TSV (word-level boxes)
TESS_CONFIG_TSV = r"--oem 1 --psm 6 -c preserve_interword_spaces=1"

def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_tsv(img: Image.Image) -> List[Dict]:
    tsv = pytesseract.image_to_data(img, lang=TESS_LANG, config=TESS_CONFIG_TSV, output_type=pytesseract.Output.DICT)
    out = []
    n = len(tsv["text"])
    for i in range(n):
        text = (tsv["text"][i] or "").strip()
        if not text:
            continue
        conf = float(tsv["conf"][i]) if tsv["conf"][i] != "-1" else -1.0
        out.append({
            "text": text,
            "left": int(tsv["left"][i]),
            "top": int(tsv["top"][i]),
            "width": int(tsv["width"][i]),
            "height": int(tsv["height"][i]),
            "conf": conf
        })
    return out

def split_columns(words: List[Dict], page_width: int) -> Tuple[List[Dict], List[Dict]]:
    # Find a robust split point: midpoint of page works well for 2-column dictionaries.
    mid = page_width / 2
    left_col, right_col = [], []
    for w in words:
        cx = w["left"] + w["width"] / 2
        if cx < mid:
            left_col.append(w)
        else:
            right_col.append(w)
    return left_col, right_col

def group_words_into_lines(words: List[Dict]) -> List[List[Dict]]:
    # Sort by y then x
    words = sorted(words, key=lambda w: (w["top"], w["left"]))

    lines: List[List[Dict]] = []
    if not words:
        return lines

    # Dynamic line height estimate
    heights = [w["height"] for w in words]
    median_h = sorted(heights)[len(heights)//2]
    # Threshold for "same line"
    y_thresh = max(8, int(median_h * 0.7))

    current = [words[0]]
    current_y = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - current_y) <= y_thresh:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x["left"]))
            current = [w]
            current_y = w["top"]

    lines.append(sorted(current, key=lambda x: x["left"]))
    return lines

def join_line(words: List[Dict]) -> str:
    # Join words with spaces; lightly clean common OCR artifacts
    s = " ".join(w["text"] for w in words)
    # Fix common OCR for asterisks
    s = s.replace(" *", " * ").replace("*  ", "* ").replace("  *", " *")
    s = " ".join(s.split())
    return s.strip()

def reflow_page(words: List[Dict], page_width: int) -> List[str]:
    left, right = split_columns(words, page_width)

    left_lines = [join_line(lw) for lw in group_words_into_lines(left)]
    right_lines = [join_line(lw) for lw in group_words_into_lines(right)]

    # Final reading order: left column top-to-bottom, then right column top-to-bottom
    # (Most dictionaries do this; if yours reads across rows, we can change later.)
    return [ln for ln in left_lines if ln] + [ln for ln in right_lines if ln]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    doc = fitz.open(INPUT_PDF)
    print(f"Pages: {doc.page_count}")

    meta = {"input_pdf": INPUT_PDF, "dpi": DPI, "lang": TESS_LANG, "psm": 6}
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for i in tqdm(range(doc.page_count), desc="OCR+reflow"):
        page = doc.load_page(i)
        zoom = DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = pixmap_to_pil(pix)

        words = ocr_tsv(img)
        lines = reflow_page(words, page_width=img.width)

        out_path = os.path.join(OUT_DIR, f"page_{i+1:04d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"Done. Reflowed OCR text saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()