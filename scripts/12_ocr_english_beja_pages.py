import os
from PIL import Image
import pytesseract
from tqdm import tqdm

IMG_DIR = "data/intermediate/en-beja/engbeja_pages_img"
OUT_DIR = "data/intermediate/en-beja/engbeja_pages_text_2col"

# Use Arabic+English if available
OCR_LANG_PRIMARY = "eng+ara"
OCR_LANG_FALLBACK = "eng"

# Column crop tuning (works for most 2-col dictionaries)
GUTTER_FRAC = 0.05
RIGHT_NUDGE_FRAC = 0.022    # was 0.015 (shift right crop LEFT by 3% of width)
MARGIN_FRAC = 0.02         # keep
LEFT_NUDGE_FRAC = 0.0      # keep

def get_lang():
    try:
        langs = pytesseract.get_languages(config="")
        if "eng" in langs and "ara" in langs:
            return OCR_LANG_PRIMARY
    except Exception:
        pass
    return OCR_LANG_FALLBACK

def crop_two_columns(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    margin = int(w * MARGIN_FRAC)
    gutter = int(w * GUTTER_FRAC)

    right_nudge = int(w * RIGHT_NUDGE_FRAC)
    left_nudge = int(w * LEFT_NUDGE_FRAC)

    mid = w // 2

    # Left column bounds
    left_x0 = max(0, margin - left_nudge)
    left_x1 = min(w, mid - gutter // 2 + left_nudge)

    # Right column bounds (shift LEFT by right_nudge, and slightly extend to the right)
    right_x0 = max(0, mid + gutter // 2 - right_nudge)
    right_x1 = min(w, w - margin + right_nudge)

    return img.crop((left_x0, 0, left_x1, h)), img.crop((right_x0, 0, right_x1, h))

def ocr(img: Image.Image, lang: str) -> list[str]:
    # PSM 6 = uniform block, good for column text
    text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)
    return lines

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
    lang = get_lang()

    for fname in tqdm(imgs, desc=f"OCR 2-col ({lang})"):
        in_path = os.path.join(IMG_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname.replace(".png", ".txt"))

        img = Image.open(in_path).convert("RGB")
        left, right = crop_two_columns(img)

        left_lines = ocr(left, lang)
        right_lines = ocr(right, lang)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {fname} col L\n")
            f.write("\n".join(left_lines) + "\n")
            f.write(f"\n# {fname} col R\n")
            f.write("\n".join(right_lines) + "\n")

    print(f"Done. Wrote column-aware OCR text to: {OUT_DIR}")

if __name__ == "__main__":
    main()