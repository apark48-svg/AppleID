"""Rule-based variety guess from color, color spread and shape.

    classify_species(apple_features(image_bgr, mask)) -> (variety, reason)

Thresholds in RULES were set on the "tune" photos in data/species_labels.csv.
"""
import cv2
import numpy as np

from allyson.detect_apple import classify_shape
from nabeeha.major_second_mask_app import analyze_mask_distribution, get_color_masks, get_hsv
from src.color import CONFIG, hue_shares

VARIETIES = {
    "granny_smith": "Granny Smith",
    "golden_delicious": "Golden Delicious",
    "red_delicious": "Red Delicious",
    "honeycrisp": "Honeycrisp",
    "gala": "Gala",
    "russet": "Russet",
}

RULES = {
    # Russet: orange-brown hue and a dark, matte skin
    "russet_hue": (8, 22),
    "russet_value": 135,
    # under this red share the apple is a green/yellow variety
    "low_red": 0.25,
    # median hue (OpenCV 0-179) at or above this is green, below is yellow
    "green_hue": 28,
    # solid red apples: red share at least this much
    "solid_red": 0.90,
    # Red Delicious is darker than the other solid red apples
    "dark_red_value": 155,
    # yellow + green share for a red blush over a yellow base (Honeycrisp)
    "blush_light": 0.45,
}


def apple_features(image_bgr, mask):
    """Measure the apple inside mask. Returns None if no usable pixels."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    valid = (mask > 0) & (hsv[:, :, 1] >= CONFIG["min_saturation"]) & (hsv[:, :, 2] >= CONFIG["min_value"])
    if np.count_nonzero(valid) < CONFIG["min_valid_pixels"]:
        return None

    shares, _ = hue_shares(image_bgr, mask)
    hue = hsv[:, :, 0][valid].astype(int)
    # red wraps around, so 170 becomes -10
    hue = np.where(hue > 90, hue - 180, hue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shape = classify_shape(max(contours, key=cv2.contourArea))

    main_color = max(shares, key=shares.get)
    main_pixels = (get_color_masks(get_hsv(image_bgr))[main_color] > 0) & (mask > 0)
    spread = analyze_mask_distribution(main_pixels) if main_pixels.any() else "not concentrated"

    return {
        "shares": shares,
        "main_color": main_color,
        "hue": float(np.median(hue)),
        "hue_spread": float(hue.std()),
        "saturation": float(np.median(hsv[:, :, 1][valid])),
        "brightness": float(np.median(hsv[:, :, 2][valid])),
        "spread": spread,
        "shape": shape,
    }


def classify_species(features, rules=RULES):
    if features is None:
        return "unknown", "no usable apple pixels"

    red = features["shares"]["red"]
    light = features["shares"]["yellow"] + features["shares"]["green"]
    hue = features["hue"]
    value = features["brightness"]

    lo, hi = rules["russet_hue"]
    if lo <= hue <= hi and value < rules["russet_value"]:
        return "russet", f"dark orange-brown skin (hue {hue:.0f}, brightness {value:.0f})"

    if red < rules["low_red"]:
        if hue >= rules["green_hue"]:
            return "granny_smith", f"green skin (hue {hue:.0f})"
        return "golden_delicious", f"yellow skin (hue {hue:.0f})"

    if (red >= rules["solid_red"] and value < rules["dark_red_value"]
            and features["spread"] == "concentrated"):
        return "red_delicious", f"solid dark red ({red:.0%} red, brightness {value:.0f})"

    if light >= rules["blush_light"]:
        return "honeycrisp", f"red blush on a yellow base ({light:.0%} yellow/green)"

    return "gala", f"mostly red with some yellow ({red:.0%} red)"
