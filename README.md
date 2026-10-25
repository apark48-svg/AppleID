# AppleID (CS 566)

Identify an apple's color (red / green / yellow) for red-green colorblind users.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Try it
```
python -m tests.smoke_test                  # synthetic apples, checks the code runs
python run_one.py data/images/<photo>.jpg   # real photo -> color, size, overlay
python run_one.py <photo> --blur kuwahara
python run_one.py <photo> --seg color       # Allyson's color-mask segmenter
python run_one.py <photo> --seg hybrid      # runs both, keeps the more apple-shaped mask
python evaluate.py                          # all segmenters on data/labels.csv -> results/
```

## Pipeline (agreed interfaces)
| Step | File | Function | Returns |
|---|---|---|---|
| Find apple | `src/segment.py` | `segment(image_bgr)` (edge), `segment_color(image_bgr)` (color, from `allyson/detect_apple.py`), or `segment_hybrid(image_bgr)` (best ellipse fit of the two) | mask (255 = apple) |
| Name color | `src/color.py` | `classify_color(image_bgr, mask)` | `(label, shares)` |
| Size | `src/size.py` | `apple_size(mask)` | width/height/aspect in px (fitEllipse) |

## Data
- Photos go in `data/images/`, named `<color>_<background>_<lighting>_<NN>.jpg`
  (e.g. `green_dark_dim_01.jpg`).
- Every photo gets a row in `data/labels.csv`:
  `filename,true_color,background,lighting,source,owner,notes`
  (`source` = self / online; `owner` = who added it; `notes` = URL for online images).
- Do not add images that are not apples.

## Adding a method (so we can compare)
- New blur: add an entry to `BLURS` in `src/segment.py`.
- New segmenter: a function with the same signature as `segment`, registered in
  `SEGMENTERS` in `src/segment.py`; `evaluate.py` and `run_one.py --seg` pick it up.
- New color method: add a function with the same signature as `classify_color`.
- Work on your own branch (`<name>-<idea>`), open a pull request into `main`.
- Put results in `results/<name>_results.csv` so the three can be compared.

## Tuning knobs
- Hue/saturation/value thresholds: `CONFIG` in `src/color.py` (OpenCV hue is 0-179).
- Plausible apple size and backup rule: `MIN_AREA_FRAC` / `MAX_AREA_FRAC` in `src/segment.py`.
