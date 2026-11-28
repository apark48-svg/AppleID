# AppleID (CS 566)

Identify an apple's color (red / green / yellow) and variety for red-green colorblind users.

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
python run_one.py <photo> --seg refined     # hybrid, then GrabCut to fix leaks / missing parts
python run_one.py <photo> --blue_mask       # Nabeeha's blue overlay of the main color
python run_one.py <photo> --seg hybrid --species --shape   # variety guess (rules) + Allyson's shape
python run_one.py <photo> --species --method model # variety guess (trained model, uses refined outlines)
python evaluate.py                          # all segmenters on data/labels.csv -> results/
python evaluate_species.py --split test     # rules on the held-out photos
python compare_species.py                   # all variety methods, lighting test, figures (~5 min)
python train_species.py                     # retrain models/species_model.joblib
python colorblind_view.py                   # how the varieties look with red-green color blindness
```

## Pipeline (agreed interfaces)
| Step | File | Function | Returns |
|---|---|---|---|
| Find apple | `src/segment.py` | `segment(image_bgr)` (edge), `segment_color(image_bgr)` (color, from `allyson/detect_apple.py`), `segment_hybrid(image_bgr)` (best ellipse fit of the two), or `segment_refined(image_bgr)` (hybrid + GrabCut) | mask (255 = apple) |
| Name color | `src/color.py` | `classify_color(image_bgr, mask)` | `(label, shares)` |
| Size | `src/size.py` | `apple_size(mask)` | width/height/aspect in px (fitEllipse) |
| Variety (rules) | `src/species.py` | `classify_species(apple_features(image_bgr, mask))` | `(variety, reason)` |
| Variety (model) | `src/classifier.py` | `predict_variety(saved, image_bgr)` | `[(variety, probability), ...]` |
| Texture | `src/texture.py` | `lbp_histogram(image_bgr, mask, radius)` | LBP histogram (10 bins) |
| Lighting | `src/lighting.py` | `gray_world(image_bgr, mask)`, `LIGHTING_CHANGES` | corrected / changed image |

## Varieties
Granny Smith, Golden Delicious, Red Delicious, Honeycrisp, Gala and Russet.
Fuji was tried first and dropped: its photos ranged from yellow-green to deep
red, and only about 3 in 10 were recognised. Russet replaced it.

Most of these are hard to tell apart for someone with red-green color
blindness: in a simulation (`colorblind_view.py`, Machado et al. 2009) Gala
and Russet become almost the same color, and Granny Smith, Honeycrisp and Gala
all turn a similar yellow-olive. Golden Delicious (yellow) and Red Delicious
(very dark) stay fairly easy. See `results/figures/colorblind_view.png` and
`results/colorblind_pairs.csv`.

### Rules (baseline)
| Variety | Rule (see `RULES` in `src/species.py`) |
|---|---|
| Russet | orange-brown hue (8-22) and dark (brightness < 135) |
| Granny Smith | little red, median hue >= 28 (green) |
| Golden Delicious | little red, median hue < 28 (yellow) |
| Red Delicious | >= 90% red, dark, main color concentrated |
| Honeycrisp | red blush, >= 45% yellow/green |
| Gala | everything else (mostly red with some yellow) |

Shape comes from Allyson's `classify_shape` and the color spread from Nabeeha's
`analyze_mask_distribution`. The rules were set on the `tune` photos only;
`evaluate_species.py --split test` checks them on photos they were not set on.

### Trained model (used by `run_one.py --species --method model`)
Logistic regression (`make_logreg`, C = 0.1) on 44 features from refined
outlines: hue histogram and color stats (plus Nabeeha's concentrated flag),
Allyson's shape numbers and LBP texture. Darker / brighter / warmer / cooler
copies of every photo are added to the training set, and a prediction is the
average over the photo's five lighting versions.

### Results
`compare_species.py` scores everything with 5-fold cross-validation repeated 5
times, keeping each photographer's photos in one fold. "Held-out" is the 66
photos the rules were not set on; "all 98" is cross-validation over every photo.

| Method | Held-out (95% interval) | All 98 |
|---|---|---|
| Rules | 55% (43-66%) | - |
| SVM: color | 60% (48-71%) | 59% |
| SVM: color + shape | 65% (53-75%) | 63% |
| SVM: color + shape + texture | 73% (61-82%) | 71% |
| SVM: + gray world | 65% (53-75%) | 67% |
| SVM: + lighting copies | 77% (65-85%) | 73% |
| SVM: refined outlines | 71% (59-81%) | 70% |
| SVM: refined + Lab + hue/sat + lighting copies | 75% (64-84%) | 72% |
| **Final model (logistic regression)** | **80% (68-88%)** | **78%** |
| CNN features (MobileNetV2), for comparison only | 87% (76-93%) | 86% |

- The final model was picked from several candidates on these same photos, so
  its number is a little optimistic. Repeating the choice inside each training
  split (nested cross-validation, `results/species_nested.csv`) gives **78%**,
  the fairest single number to report.
- The right variety is in the final model's top two guesses 93% of the time.
- When the photos are made darker, brighter, warmer or cooler, the final model
  gets 61-78%; the rules drop to 29% on cooler photos. Gray world hurts because
  many photos have white, black or colored backgrounds, which breaks its "the
  scene is gray on average" assumption.
- The CNN row uses a pretrained MobileNetV2 as a feature extractor
  (`src/cnn.py`). It is not part of the app; it only shows how far learned
  features go. It appears only if `models/mobilenetv2-12.onnx` is present:
  `curl -L -o models/mobilenetv2-12.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx`

Outlines: `segment_refined` (hybrid + GrabCut, `--seg refined`) gave a correct
outline on 74 of the 98 variety photos, against 65 for `segment_hybrid`
(checked by eye, `results/outline_check.csv`). The rest are mostly apples on
backgrounds of a similar color (leaves, a wooden table, a towel, a plate).

Figures are in `results/figures/`, numbers in `results/species_*.csv`.

## Data
- Photos go in `data/images/`, named `<color>_<background>_<lighting>_<NN>.jpg`
  (e.g. `green_dark_dim_01.jpg`).
- Every photo gets a row in `data/labels.csv`:
  `filename,true_color,background,lighting,source,owner,notes`
  (`source` = self / online; `owner` = who added it; `notes` = URL for online images).
- Do not add images that are not apples.
- Variety photos are in `data/species/` (98 photos), listed in `data/species_labels.csv`:
  `filename,variety,split,group,source,license,author,note`. They are from
  Wikimedia Commons and Flickr (found through Openverse); the source page,
  license and author of each one are in that file. Some licenses are
  non-commercial (CC BY-NC), which is fine for a class project but not for a
  product. Some Golden Delicious and Russet photos are crops of single apples
  from photos with several apples (`note` says where).
- `split` is `tune` (used to set the rules) or `test` (held out). `group` is
  the photographer or source photo; cross-validation keeps a group in one fold.

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
