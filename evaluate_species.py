"""Accuracy of the variety rules on data/species_labels.csv.

    python evaluate_species.py [--split test]
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.segment import SEGMENTERS
from src.species import VARIETIES, apple_features, classify_species


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test", "all"], default="all")
    ap.add_argument("--seg", choices=list(SEGMENTERS), default="hybrid")
    args = ap.parse_args()

    labels = pd.read_csv("data/species_labels.csv")
    if args.split != "all":
        labels = labels[labels.split == args.split]

    rows = []
    for _, r in labels.iterrows():
        img = cv2.imread(f"data/species/{r.filename}")
        if img is None:
            print(f"{r.filename}: could not read")
            continue
        mask = SEGMENTERS[args.seg](img)
        feats = apple_features(img, mask)
        guess, reason = classify_species(feats)
        row = {"filename": r.filename, "split": r.split, "variety": r.variety, "guess": guess,
               "correct": guess == r.variety, "reason": reason}
        if feats:
            row.update({f"share_{c}": round(v, 3) for c, v in feats["shares"].items()})
            row.update({"hue": feats["hue"], "brightness": feats["brightness"], "spread": feats["spread"],
                        "proportion": feats["shape"]["proportion"], "symmetry": feats["shape"]["symmetry"]})
        rows.append(row)

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / "species_results.csv"
    df.to_csv(out, index=False)

    for split, part in df.groupby("split", sort=False):
        print(f"\n== {split}: {part.correct.sum()}/{len(part)} correct")
        for v in VARIETIES:
            p = part[part.variety == v]
            if len(p):
                print(f"   {VARIETIES[v]:17s} {p.correct.sum()}/{len(p)}")
        for _, w in part[~part.correct].iterrows():
            print(f"   WRONG {w.filename}: guessed {w.guess} ({w.reason})")

    order = list(VARIETIES) + ["unknown"]
    table = pd.crosstab(df.variety, df.guess).reindex(index=list(VARIETIES), columns=order, fill_value=0)
    table = table.loc[:, (table != 0).any()]
    print("\nconfusion (rows = true, columns = guess):")
    print(table.to_string())
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
