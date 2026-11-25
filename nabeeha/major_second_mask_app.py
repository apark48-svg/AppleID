import os
import sys
import cv2
import numpy as np

COLOR_RANGES = {
    "red": [((0, 70, 40), (8, 255, 255)), ((160, 70, 40), (179, 255, 255))],
    "yellow": [((9, 70, 40), (22, 255, 255))],
    "green": [((23, 50, 40), (85, 255, 255))],
}

DEFAULT_IMAGES_DIRS = [
    r"AppleID\poorvajha\data\images",
    r"AppleID\nabeeha\dataset",
]
DEFAULT_OUTPUT_DIR = r"AppleID\nabeeha\output_masked"
DEFAULT_REPORT_FILE = "concentration.txt"

ROYAL_BLUE = np.array([225, 105, 65], dtype=np.uint8)


def get_hsv(img, ksize=7):
    moving_avg = cv2.boxFilter(img, ddepth=-1, ksize=(ksize, ksize), normalize=True)
    return cv2.cvtColor(moving_avg, cv2.COLOR_BGR2HSV)


def get_color_masks(hsv):
    masks = {}
    for color, ranges in COLOR_RANGES.items():
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            m = m | cv2.inRange(hsv, np.array(lo), np.array(hi))
        masks[color] = m
    return masks


def get_apple_mask(hsv, color_masks):
    _, sat = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    colors = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for m in color_masks.values():
        colors = colors | m
    mask = sat & colors
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)


def find_apples(mask, min_frac=0.01, max_ratio=1.5):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = mask.shape[0] * mask.shape[1] * min_frac
    apples = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area and len(c) >= 5:
            (_, _), (a, b), _ = cv2.fitEllipse(c)
            if min(a, b) > 0 and max(a, b) / min(a, b) <= max_ratio:
                apples.append((area, c))
    apples.sort(key=lambda item: item[0], reverse=True)
    return [c for _, c in apples]


def classify_pixel_hsv(h, s, v):
    for color, ranges in COLOR_RANGES.items():
        for (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v) in ranges:
            if lo_h <= h <= hi_h and lo_s <= s <= hi_s and lo_v <= v <= hi_v:
                return color
    return None


def identify_pixels_and_majority_mask(hsv, contours):
    h, w = hsv.shape[:2]
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(region, contours, -1, 255, -1)

    pixel_labels = np.empty((h, w), dtype=object)
    pixel_labels[:] = None
    counts = {color: 0 for color in COLOR_RANGES}

    ys, xs = np.where(region > 0)
    for y, x in zip(ys, xs):
        label = classify_pixel_hsv(int(hsv[y, x, 0]), int(hsv[y, x, 1]), int(hsv[y, x, 2]))
        pixel_labels[y, x] = label
        if label:
            counts[label] += 1

    majority_color = max(counts, key=counts.get)
    return majority_color, pixel_labels == majority_color


def analyze_mask_distribution(mask_bool, dominant_share_threshold=0.70, min_spot_pixels=15):
    total_pixels = int(np.count_nonzero(mask_bool))
    mask_u8 = mask_bool.astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    component_areas = sorted(
        (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)),
        reverse=True,
    )
    largest_share = component_areas[0] / total_pixels
    significant_spots = sum(1 for area in component_areas if area >= min_spot_pixels)

    ys, xs = np.where(mask_bool)
    bbox_area = (int(xs.max() - xs.min() + 1)) * (int(ys.max() - ys.min() + 1))
    fill_density = total_pixels / bbox_area

    is_concentrated = (
        largest_share >= dominant_share_threshold
        and fill_density >= 0.35
        and significant_spots <= 2
    )
    return "concentrated" if is_concentrated else "not concentrated"


def mask_majority_pixel_color(img, alpha=0.6):
    hsv = get_hsv(img)
    color_masks = get_color_masks(hsv)
    apple_mask = get_apple_mask(hsv, color_masks)
    contours = find_apples(apple_mask)

    _, majority_bool = identify_pixels_and_majority_mask(hsv, contours)
    concentration_type = analyze_mask_distribution(majority_bool)

    overlay = img.copy()
    overlay[majority_bool] = ROYAL_BLUE
    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    out = img.copy()
    out[majority_bool] = blended[majority_bool]
    return out, concentration_type


def process_directories(
    images_dirs=DEFAULT_IMAGES_DIRS,
    output_dir=DEFAULT_OUTPUT_DIR,
    report_filename=DEFAULT_REPORT_FILE,
    alpha=0.6,
):
    os.makedirs(output_dir, exist_ok=True)
    lines = []

    for images_dir in images_dirs:
        image_files = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_result" not in f
        )
        for name in image_files:
            img = cv2.imread(os.path.join(images_dir, name))
            out, concentration_type = mask_majority_pixel_color(img, alpha=alpha)
            cv2.imwrite(os.path.join(output_dir, f"{name.rsplit('.', 1)[0]}.png"), out)
            lines.append(f"{name}: {concentration_type}")

    with open(os.path.join(output_dir, report_filename), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    in_dirs = sys.argv[1:3] if len(sys.argv) >= 3 else DEFAULT_IMAGES_DIRS
    out_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_DIR
    process_directories(in_dirs, out_dir)