import os
import cv2
import numpy as np
from tqdm import tqdm

IN_DIR = "data/intermediate/beja2/dict2_pages_img"
OUT_DIR = "data/intermediate/beja2/dict2_pages_pre"

def deskew(img: np.ndarray) -> np.ndarray:
    # light deskew using minAreaRect on thresholded pixels
    coords = np.column_stack(np.where(img < 200))
    if len(coords) < 1000:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Contrast boost (great for faded print)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise while keeping edges
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Adaptive threshold (handles uneven fading)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

    # Remove tiny scribbles/noise (tunable)
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    # Deskew after threshold (optional but helps)
    clean = deskew(clean)
    return clean

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(IN_DIR) if f.endswith(".png")])

    for f in tqdm(files, desc="Preprocessing pages"):
        inp = cv2.imread(os.path.join(IN_DIR, f))
        out = preprocess(inp)
        cv2.imwrite(os.path.join(OUT_DIR, f), out)

    print(f"Preprocessed pages -> {OUT_DIR}")

if __name__ == "__main__":
    main()