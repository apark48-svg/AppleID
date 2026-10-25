"""Print hue statistics for every labeled image, to tune the hue ranges.

Save this in the appleid/ folder (next to run_one.py) and run:
    python hue_stats.py
Paste the printed table back into the chat.
"""
import cv2
import numpy as np
import pandas as pd

from src.color import CONFIG, classify_color
from src.segment import segment

rows = pd.read_csv("data/labels.csv")
print(f"{'file':26s} {'true':6s} {'pred':7s} {'seg':8s} {'hue p10/med/p90':16s} {'medS':>4s} {'medV':>4s}  shares(r/y/g)")
for _, r in rows.iterrows():
    img = cv2.imread(f"data/images/{r.filename}")
    if img is None:
        print(f"{r.filename}: could not read")
        continue
    mask, method = segment(img, return_method=True)
    label, shares = classify_color(img, mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    valid = (mask > 0) & (hsv[:, :, 1] >= CONFIG["min_saturation"]) & (hsv[:, :, 2] >= CONFIG["min_value"])
    if valid.sum() == 0:
        print(f"{r.filename:26s} {r.true_color:6s} {label:7s} {method:8s} no valid pixels")
        continue
    h = hsv[:, :, 0][valid]; s = hsv[:, :, 1][valid]; v = hsv[:, :, 2][valid]
    p10, med, p90 = np.percentile(h, [10, 50, 90])
    print(f"{r.filename:26s} {r.true_color:6s} {label:7s} {method:8s} "
          f"{p10:4.0f}/{med:3.0f}/{p90:3.0f}      {np.median(s):4.0f} {np.median(v):4.0f}  "
          f"{shares['red']:.2f}/{shares['yellow']:.2f}/{shares['green']:.2f}")
