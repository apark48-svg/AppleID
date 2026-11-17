"""Try the pipeline on one photo:

    python run_one.py data/images/green_light_good_01.jpg
    python run_one.py data/images/green_light_good_01.jpg --seg color

Prints the color verdict, size and position, and saves an annotated image to
results/overlays/ (outline, bounding box, center, color label).
"""
import argparse
from pathlib import Path

import cv2

from src.color import classify_color
from src.segment import SEGMENTERS, segment
from src.size import apple_size
from src.visualize import annotate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--blur", default="gaussian", choices=["gaussian", "kuwahara"],
                    help="blur for the edge segmenter")
    ap.add_argument("--seg", default="edge", choices=list(SEGMENTERS),
                    help="edge = Canny/GrabCut (src), color = Allyson's color-mask method")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    if args.seg == "edge":
        mask, method = segment(img, blur=args.blur, return_method=True)
    else:
        mask, method = SEGMENTERS[args.seg](img, return_method=True)
    label, shares = classify_color(img, mask)
    size = apple_size(mask)

    out = Path("results/overlays") / f"{Path(args.image).stem}_{args.seg}_overlay.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotate(img, mask, label, shares, size))

    print(f"segmentation: {method}")
    print(f"color: {label}  shares: { {k: round(v, 2) for k, v in shares.items()} }")
    if size:
        cx, cy = size["center_xy"]
        print(f"position: center=({cx:.0f}, {cy:.0f})  bbox(x,y,w,h)={size['bbox_xywh']}  "
              f"image={img.shape[1]}x{img.shape[0]}")
        print(f"size: width={size['width_px']:.0f}px height={size['height_px']:.0f}px "
              f"aspect={size['aspect']:.2f}")
    else:
        print("size: could not fit an ellipse")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
