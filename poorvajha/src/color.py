"""Step 4: name the apple's color with HSV hue-range masking.

Public interface (agreed with the team):
    classify_color(image_bgr, mask) -> (label, shares)
    label  : "red" | "green" | "yellow" | "unknown"
    shares : {"red": 0.12, "green": 0.80, "yellow": 0.08}  (sums to ~1 over
             the pixels that fell in any range)

OpenCV hue runs 0-179 (not 0-359). Red wraps around, so it needs two ranges.
All thresholds live in CONFIG so you can tune them without touching the logic.
"""
import cv2
import numpy as np

CONFIG = {
    # (low, high) inclusive hue ranges, OpenCV scale 0-179

"hue_ranges": {
    "red": [(0, 12), (168, 179)],
    "yellow": [(13, 21)],
    "green": [(22, 85)],
},

    # Ignore washed-out / near-black pixels: their hue is unreliable
    # (specular highlights, deep shadow, gray background bleed).
    "min_saturation": 40,
    "min_value": 30,
    # If fewer than this many apple pixels survive the filters, say "unknown".
    "min_valid_pixels": 50,
}


def _range_masks(image_bgr, mask, config):
    """Return ({color: bool array}, valid bool array) for pixels in the mask."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    valid = (mask > 0) & (s >= config["min_saturation"]) & (v >= config["min_value"])
    out = {}
    for color, ranges in config["hue_ranges"].items():
        in_range = np.zeros_like(valid)
        for lo, hi in ranges:
            in_range |= (h >= lo) & (h <= hi)
        out[color] = valid & in_range
    return out, valid


def hue_shares(image_bgr, mask, config=CONFIG):
    """Return ({color: share}, n_valid_pixels) for pixels inside the mask."""
    masks, valid = _range_masks(image_bgr, mask, config)
    n_valid = int(np.count_nonzero(valid))
    counts = {color: int(np.count_nonzero(m)) for color, m in masks.items()}

    total = sum(counts.values())
    if total == 0:
        return {c: 0.0 for c in counts}, n_valid
    return {c: counts[c] / total for c in counts}, n_valid


def classify_color(image_bgr, mask, config=CONFIG):
    shares, n_valid = hue_shares(image_bgr, mask, config)
    if n_valid < config["min_valid_pixels"] or max(shares.values()) == 0:
        return "unknown", shares
    return max(shares, key=shares.get), shares
