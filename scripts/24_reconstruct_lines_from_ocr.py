import json
import os
from typing import List, Dict, Any

IN_PATH = "data/intermediate/beja2/dict2_pages_ocr_data.jsonl"
OUT_LINES = "data/intermediate/beja2/dict2_reconstructed_lines.jsonl"

def main():
    if os.path.exists(OUT_LINES):
        os.remove(OUT_LINES)

    with open(IN_PATH, "r", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]

    for page in pages:
        words = page["words"]
        # Sort by y then x
        words.sort(key=lambda w: (w["y"], w["x"]))

        # Adaptive line threshold based on median word height
        hs = sorted([w["h"] for w in words if w["h"] > 0])
        median_h = hs[len(hs)//2] if hs else 20
        y_thresh = max(10, int(median_h * 0.7))

        lines: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_y = None

        for w in words:
            if current_y is None:
                current = [w]
                current_y = w["y"]
                continue

            # if far enough in y, start new line
            if abs(w["y"] - current_y) > y_thresh:
                lines.append(current)
                current = [w]
                current_y = w["y"]
            else:
                current.append(w)
                # track running average y to tolerate slant/curvature
                current_y = int(sum(x["y"] for x in current) / len(current))

        if current:
            lines.append(current)

        # Convert each line to text (sort by x within line)
        for idx, line_words in enumerate(lines, start=1):
            line_words.sort(key=lambda w: w["x"])
            text = " ".join(w["t"] for w in line_words).strip()
            conf = sum(w["c"] for w in line_words) / len(line_words) if line_words else 0.0

            rec = {
                "page": page["page"],
                "line_index": idx,
                "text": text,
                "avg_conf": conf,
                "psm": page["psm"],
                "source": {
                    "image": page["image"],
                }
            }
            with open(OUT_LINES, "a", encoding="utf-8") as out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote reconstructed lines -> {OUT_LINES}")

if __name__ == "__main__":
    main()