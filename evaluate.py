"""Run every segmenter on every labeled photo and compare them.

    python evaluate.py              # all segmenters in SEGMENTERS
    python evaluate.py --seg color  # just one

Writes results/<segmenter>_results.csv (one row per photo) and prints color
accuracy per segmenter. Also saves overlays to results/overlays/<segmenter>/
so failures can be checked by eye.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.color import classify_color
from src.segment import SEGMENTERS, save_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", choices=list(SEGMENTERS), help="default: all")
    args = ap.parse_args()
    names = [args.seg] if args.seg else list(SEGMENTERS)

    labels = pd.read_csv("data/labels.csv")
    Path("results").mkdir(exist_ok=True)

    for name in names:
        seg = SEGMENTERS[name]
        overlays = Path("results/overlays") / name
        overlays.mkdir(parents=True, exist_ok=True)
        rows = []
        for _, r in labels.iterrows():
            img = cv2.imread(f"data/images/{r.filename}")
            if img is None:
                print(f"{r.filename}: could not read")
                continue
            mask, method = seg(img, return_method=True)
            label, shares = classify_color(img, mask)
            save_overlay(img, mask, overlays / r.filename)
            rows.append({
                "filename": r.filename,
                "true_color": r.true_color,
                "pred_color": label,
                "correct": label == r.true_color,
                "seg_method": method,
                "mask_frac": round(float(np.count_nonzero(mask)) / mask.size, 3),
                **{f"share_{c}": round(v, 3) for c, v in shares.items()},
                "background": r.background,
                "lighting": r.lighting,
            })
        df = pd.DataFrame(rows)
        out = Path("results") / f"{name}_results.csv"
        df.to_csv(out, index=False)
        print(f"{name:6s} color accuracy: {df.correct.sum()}/{len(df)}  -> {out}")
        for _, w in df[~df.correct].iterrows():
            print(f"    WRONG {w.filename}: predicted {w.pred_color}")


if __name__ == "__main__":
    main()
