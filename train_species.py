"""Train the final variety model on all photos and save models/species_model.joblib.

    python train_species.py
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from src.classifier import FINAL_PREFIXES, feature_vector, make_logreg
from src.lighting import LIGHTING_CHANGES
from src.segment import segment_refined

MODEL_PATH = Path("models/species_model.joblib")


def main():
    labels = pd.read_csv("data/species_labels.csv")
    rows, y = [], []
    for _, r in labels.iterrows():
        original = cv2.imread(f"data/species/{r.filename}")
        for change in LIGHTING_CHANGES.values():
            img = change(original)
            f = feature_vector(img, segment_refined(img))
            if f is None:
                continue
            rows.append({k: v for k, v in f.items() if k.startswith(FINAL_PREFIXES)})
            y.append(r.variety)
    X = pd.DataFrame(rows).fillna(0.0)

    model = make_logreg().fit(X.values, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)
    print(f"trained on {len(labels)} photos x {len(LIGHTING_CHANGES)} lighting versions "
          f"({len(X.columns)} features, varieties: {', '.join(np.unique(y))}), saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
