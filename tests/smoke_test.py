"""Smoke test on SYNTHETIC apples (colored ellipses). It checks that the code
runs and that the obvious cases work. It says nothing about real-photo accuracy.

    python -m tests.smoke_test
"""
import itertools
from pathlib import Path

import cv2
import numpy as np

from src.color import classify_color
from src.segment import segment
from src.size import apple_size

OUT = Path("tests/_synthetic")
OUT.mkdir(parents=True, exist_ok=True)

# BGR colors for a clearly red / green / yellow apple
COLORS = {"red": (30, 30, 190), "green": (50, 170, 60), "yellow": (40, 200, 220)}
BGS = {"dark": 15, "light": 235}
LIGHT = {"dim": 0.45, "good": 1.0}


def make(color, bg, light, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((600, 800, 3), BGS[bg], np.uint8)
    apple = np.zeros((600, 800), np.uint8)
    cv2.ellipse(apple, (400, 300), (150, 170), 10, 0, 360, 255, -1)
    # simple shading so the apple is not a flat color
    yy, xx = np.mgrid[0:600, 0:800]
    shade = 1.0 - 0.35 * (np.hypot(xx - 350, yy - 250) / 250).clip(0, 1)
    fill = (np.array(COLORS[color], np.float32)[None, None, :] * shade[..., None])
    img[apple > 0] = fill[apple > 0].astype(np.uint8)
    img = (img.astype(np.float32) * LIGHT[light]).clip(0, 255)
    img = img + rng.normal(0, 4, img.shape)
    return img.clip(0, 255).astype(np.uint8), apple


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max((a | b).sum(), 1)


def main():
    correct = total = 0
    for blur in ("gaussian", "kuwahara"):
        print(f"\n--- blur = {blur} ---")
        for color, bg, light in itertools.product(COLORS, BGS, LIGHT):
            img, truth = make(color, bg, light)
            mask, method = segment(img, blur=blur, return_method=True)
            label, shares = classify_color(img, mask)
            size = apple_size(mask)
            ok = label == color
            correct += ok
            total += 1
            print(f"{color:6s} {bg:5s} {light:4s} -> {label:7s} "
                  f"{'OK ' if ok else 'BAD'} seg={method:8s} IoU={iou(mask, truth):.2f} "
                  f"aspect={size['aspect']:.2f}")
            if blur == "gaussian":
                cv2.imwrite(str(OUT / f"{color}_{bg}_{light}.jpg"), img)
    print(f"\ncolor correct: {correct}/{total}")


if __name__ == "__main__":
    main()
