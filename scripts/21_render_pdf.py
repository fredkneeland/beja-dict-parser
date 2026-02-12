import os
import fitz  # pymupdf
from tqdm import tqdm

PDF_PATH = "data/input/dict2.pdf"  # <-- change to your filename
OUT_DIR = "data/intermediate/beja2/dict2_pages_img"
DPI = 450  # quality > speed

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = fitz.open(PDF_PATH)

    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in tqdm(range(len(doc)), desc="Rendering pages"):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(OUT_DIR, f"page_{i+1:04d}.png")
        pix.save(out_path)

    print(f"Rendered {len(doc)} pages -> {OUT_DIR}")

if __name__ == "__main__":
    main()