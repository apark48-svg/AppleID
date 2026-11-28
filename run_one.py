"""Run the pipeline on one photo.

    python run_one.py data/images/green_light_good_01.jpg
    python run_one.py data/species/gala_01.jpg --blue_mask --shape --species --method model
"""
import argparse
from pathlib import Path

import cv2

from src.color import classify_color
from src.segment import SEGMENTERS, segment
from src.size import apple_size
from src.species import VARIETIES, apple_features, classify_species
from src.visualize import annotate


def model_species(img, path="models/species_model.joblib"):
    import joblib
    from src.classifier import predict_variety

    if not Path(path).exists():
        return None, f"no model at {path}, run python train_species.py first"
    ranked = predict_variety(joblib.load(path), img)
    (best, p1), (second, p2) = ranked[0], ranked[1]
    name = VARIETIES.get(best, best)
    return name, f"{name}  ({p1:.0%} sure; next closest: {VARIETIES.get(second, second)} {p2:.0%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--blur", default="gaussian", choices=["gaussian", "kuwahara"],
                    help="blur for the edge segmenter")
    ap.add_argument("--seg", default="edge", choices=list(SEGMENTERS),
                    help="edge = Canny/GrabCut (src), color = Allyson's color-mask method")
    ap.add_argument("--blue_mask", action="store_true",
                    help="save Nabeeha's blue overlay of the main color")
    ap.add_argument("--shape", action="store_true", help="print Allyson's shape description")
    ap.add_argument("--species", action="store_true", help="guess the apple variety")
    ap.add_argument("--method", default="rules", choices=["rules", "model"],
                    help="how --species decides (model needs models/species_model.joblib)")
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

    if args.shape or args.species:
        feats = apple_features(img, mask)
        if feats is None:
            print("shape/species: not enough apple pixels")
        else:
            if args.shape:
                print(f"shape: {feats['shape']}")
            if args.species and args.method == "rules":
                variety, reason = classify_species(feats)
                variety = VARIETIES.get(variety, variety)
                print(f"species: {variety}  ({reason})")
            elif args.species:
                variety, answer = model_species(img)
                print(f"species: {answer}")
            if args.species and variety in VARIETIES.values():
                first, second = sorted(shares, key=shares.get, reverse=True)[:2]
                skin = f"mostly {first}" + (f" with some {second}" if shares[second] >= 0.2 else "")
                print(f"answer: This looks like a {variety} apple. Its skin is {skin}.")

    if args.blue_mask:
        from nabeeha.major_second_mask_app import mask_majority_pixel_color
        blue, spread = mask_majority_pixel_color(img)
        blue_out = Path("results/overlays") / f"{Path(args.image).stem}_blue_mask.png"
        cv2.imwrite(str(blue_out), blue)
        print(f"main color spread: {spread}")
        print(f"saved: {blue_out}")


if __name__ == "__main__":
    main()
