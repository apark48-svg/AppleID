"""Skin texture as a uniform LBP histogram (10 bins).

    lbp_histogram(image_bgr, mask, radius=1)
"""
import cv2
import numpy as np

SIZE = 128


def _uniform_lookup():
    table = np.zeros(256, np.uint8)
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        changes = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        table[code] = sum(bits) if changes <= 2 else 9
    return table


UNIFORM = _uniform_lookup()


def lbp_codes(gray, radius=1):
    g = gray.astype(np.int16)
    h, w = g.shape
    padded = np.pad(g, radius, mode="edge")
    codes = np.zeros((h, w), np.uint8)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dy = int(round(-radius * np.sin(angle)))
        dx = int(round(radius * np.cos(angle)))
        neighbour = padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
        codes |= (neighbour >= g).astype(np.uint8) << i
    return UNIFORM[codes]


def lbp_histogram(image_bgr, mask, radius=1):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros(10)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    inside = cv2.resize(mask[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
    inside = cv2.erode(inside.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if not inside.any():
        return np.zeros(10)
    labels = lbp_codes(gray, radius)[inside]
    hist = np.bincount(labels, minlength=10).astype(float)
    return hist / hist.sum()
