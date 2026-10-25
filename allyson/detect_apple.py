import sys
import cv2
import numpy as np

# opencv hsv: h is 0-179, s and v are 0-255
# red wraps around 0 so it gets two ranges
# 11-19 is orange blush, counted as red (move it to yellow if your yellow apples come out red)

# greens output as yellow, yellow output as green
# COLOR_RANGES = {
#     "red": [((0, 70, 40), (19, 255, 255)), ((160, 70, 40), (179, 255, 255))],
#     "yellow": [((20, 70, 40), (34, 255, 255))],
#     "green": [((35, 50, 40), (85, 255, 255))],
# }
COLOR_RANGES = {
    "red": [((0, 70, 40), (8, 255, 255)), ((160, 70, 40), (179, 255, 255))],
    "yellow": [((9, 70, 40), (22, 255, 255))],
    "green": [((23, 50, 40), (85, 255, 255))],
}


def load_image(filename):
    img = cv2.imread(filename)
    if img is None:
        print("couldnt read", filename)
        sys.exit(1)
    return img


def get_hsv(img):
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


def get_color_masks(hsv):
    masks = {}
    for color in COLOR_RANGES:
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in COLOR_RANGES[color]:
            m = m | cv2.inRange(hsv, np.array(lo), np.array(hi))
        masks[color] = m
    return masks


def get_apple_mask(hsv, color_masks):
    # saturation (otsu) works on dull backgrounds, color masks work on colorful ones
    # keep pixels that pass both so wood tables and blue cloths both get cut out
    _, sat = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    colors = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color in color_masks:
        colors = colors | color_masks[color]

    mask = sat & colors

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


def find_apple(mask, min_frac=0.01, max_ratio=1.5):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = mask.shape[0] * mask.shape[1] * min_frac

    best = None
    best_ellipse = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 5:  # fitEllipse needs 5+ points
            continue

        ellipse = cv2.fitEllipse(c)
        (_, _), (a, b), _ = ellipse
        if min(a, b) <= 0:
            continue

        # apples are close to round
        ratio = max(a, b) / min(a, b)
        if ratio > max_ratio:
            continue

        if area > best_area:
            best_area = area
            best = c
            best_ellipse = ellipse

    return best, best_ellipse


def classify_color(contour, color_masks):
    apple = np.zeros(color_masks["red"].shape, dtype=np.uint8)
    cv2.drawContours(apple, [contour], -1, 255, -1)

    counts = {}
    for color in color_masks:
        counts[color] = cv2.countNonZero(color_masks[color] & apple)

    total = sum(counts.values())
    if total == 0:
        return None, counts

    best = max(counts, key=counts.get)
    pcts = {}
    for color in counts:
        pcts[color] = round(100 * counts[color] / total, 1)
    return best, pcts


def draw_result(img, contour, ellipse, color):
    out = img.copy()
    cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)
    cv2.ellipse(out, ellipse, (255, 0, 0), 2)

    (cx, cy), _, _ = ellipse
    label = color + " apple"
    cv2.putText(out, label, (int(cx) - 70, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(out, label, (int(cx) - 70, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return out


def detect_apple(img):
    hsv = get_hsv(img)
    color_masks = get_color_masks(hsv)
    mask = get_apple_mask(hsv, color_masks)

    contour, ellipse = find_apple(mask)
    if contour is None:
        return None, None, img.copy()

    color, pcts = classify_color(contour, color_masks)
    if color is None:
        return None, None, img.copy()

    return color, pcts, draw_result(img, contour, ellipse, color)


if __name__ == "__main__":
    import os
    folder = sys.argv[1]
    letters = {"r": "red", "g": "green", "y": "yellow"}

    correct = 0
    total = 0
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")) or "_result" in name:
            continue

        path = os.path.join(folder, name)
        img = load_image(path)
        color, pcts, out = detect_apple(img)
        cv2.imwrite(path.rsplit(".", 1)[0] + "_result.png", out)

        expected = letters.get(name[0].lower())
        if expected is None:
            print(name, "->", color, pcts)
            continue

        total += 1
        if color == expected:
            correct += 1
            print(name, "->", color, "ok")
        else:
            print(name, "->", color, "WRONG (expected", expected + ")", pcts)

    if total > 0:
        print("accuracy:", correct, "/", total, "=", round(100 * correct / total, 1), "%")
