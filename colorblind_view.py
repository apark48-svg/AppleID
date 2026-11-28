"""Simulate red-green color blindness (Machado et al., 2009) on the six varieties
and measure how similar their colors become.

    python colorblind_view.py
"""
import itertools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.segment import segment_refined
from src.species import VARIETIES

SIMULATIONS = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
}


def _to_linear(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255


def simulate(image_bgr, kind):
    rgb = _to_linear(image_bgr[..., ::-1].astype(float))
    out = _to_srgb(rgb @ SIMULATIONS[kind].T)
    return out[..., ::-1].astype(np.uint8)


def mean_lab(image_bgr, mask):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[mask].astype(float).mean(axis=0)
    # OpenCV stores L as 0-255 and a, b shifted by 128
    return lab * [100 / 255, 1, 1] - [0, 128, 128]


def main():
    labels = pd.read_csv("data/species_labels.csv")
    views = ["normal"] + list(SIMULATIONS)
    means, examples = {v: {} for v in views}, {}
    for variety in VARIETIES:
        colors = {v: [] for v in views}
        for name in labels[labels.variety == variety].filename:
            img = cv2.imread(f"data/species/{name}")
            mask = segment_refined(img) > 0
            examples.setdefault(variety, img)
            for v in views:
                colors[v].append(mean_lab(img if v == "normal" else simulate(img, v), mask))
        for v in views:
            means[v][variety] = np.mean(colors[v], axis=0)

    rows = []
    for a, b in itertools.combinations(VARIETIES, 2):
        rows.append({"pair": f"{VARIETIES[a]} vs {VARIETIES[b]}",
                     **{v: round(float(np.linalg.norm(means[v][a] - means[v][b])), 1) for v in views}})
    pairs = pd.DataFrame(rows).sort_values("deuteranopia")
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    pairs.to_csv("results/colorblind_pairs.csv", index=False)
    print(pairs.to_string(index=False))

    tiles = []
    for variety, img in examples.items():
        k = 200 / max(img.shape[:2])
        top = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        bottom = cv2.resize(simulate(img, "deuteranopia"), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        tile = np.full((450, 210, 3), 255, np.uint8)
        tile[:top.shape[0], :top.shape[1]] = top
        tile[210:210 + bottom.shape[0], :bottom.shape[1]] = bottom
        cv2.putText(tile, VARIETIES[variety], (4, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.hstack(tiles)
    header = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(header, "top: normal vision   bottom: deuteranopia (simulated)", (4, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite("results/figures/colorblind_view.png", np.vstack([header, sheet]))
    print("saved results/figures/colorblind_view.png and results/colorblind_pairs.csv")


if __name__ == "__main__":
    main()
