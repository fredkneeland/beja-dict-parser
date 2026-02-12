import os
import json
import cv2
import pytesseract
from tqdm import tqdm

IN_DIR = "data/intermediate/dict2_pages_pre"
OUT_PATH = "data/intermediate/dict2_pages_ocr_data.jsonl"

def norm(s: str) -> str:
    s = s.replace("‎", "").replace("\u200f", "").replace("\u200e", "")
    return " ".join(s.split()).strip()

def main():
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    pages = sorted([f for f in os.listdir(IN_DIR) if f.endswith(".png")])

    # Try multiple PSMs; keep the one with best average confidence
    psms = [6, 4, 11]  # 6=block, 4=column-ish, 11=sparse text

    for pf in tqdm(pages, desc="OCR full pages"):
        page_num = int(pf.split("_")[1].split(".")[0])
        img_path = os.path.join(IN_DIR, pf)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # upscale a bit
        img = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)

        best = None

        for psm in psms:
            cfg = f"--oem 1 --psm {psm}"
            data = pytesseract.image_to_data(
                img, lang="eng", config=cfg, output_type=pytesseract.Output.DICT
            )

            words = []
            confs = []
            n = len(data["text"])
            for i in range(n):
                txt = norm(data["text"][i])
                if not txt:
                    continue
                try:
                    conf = float(data["conf"][i])
                except Exception:
                    continue
                if conf < 0:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                words.append({"t": txt, "c": conf, "x": x, "y": y, "w": w, "h": h})
                confs.append(conf)

            avg_conf = sum(confs) / len(confs) if confs else 0.0
            candidate = {"psm": psm, "avg_conf": avg_conf, "words": words}

            if best is None or candidate["avg_conf"] > best["avg_conf"]:
                best = candidate

        out = {
            "page": page_num,
            "image": img_path,
            "psm": best["psm"],
            "avg_conf": best["avg_conf"],
            "words": best["words"],
        }

        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote OCR data -> {OUT_PATH}")

if __name__ == "__main__":
    main()