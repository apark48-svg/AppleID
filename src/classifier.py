"""Features and models for the learned variety classifier.

Feature groups: color_, shape_ (Allyson's classify_shape), texture_ (LBP),
lab_ and hs_. The saved model uses FINAL_PREFIXES.
"""
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.lighting import LIGHTING_CHANGES, gray_world
from src.segment import segment_refined
from src.species import apple_features
from src.texture import lbp_histogram

HUE_BINS = 12
FINAL_PREFIXES = ("color_", "shape_", "texture_")
HS_BINS = (8, 3)


def feature_vector(image_bgr, mask, lighting=False):
    if lighting:
        image_bgr = gray_world(image_bgr, mask)
    f = apple_features(image_bgr, mask)
    if f is None:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    valid = (mask > 0) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 30)
    hue_hist = np.histogram(hsv[:, :, 0][valid], bins=HUE_BINS, range=(0, 180))[0].astype(float)
    hue_hist /= max(hue_hist.sum(), 1)

    out = {f"color_hue_{i:02d}": v for i, v in enumerate(hue_hist)}
    out.update({f"color_share_{c}": v for c, v in f["shares"].items()})
    out.update({
        "color_hue_median": f["hue"],
        "color_hue_spread": f["hue_spread"],
        "color_saturation": f["saturation"],
        "color_brightness": f["brightness"],
        "color_concentrated": float(f["spread"] == "concentrated"),
    })
    for k in ("circularity", "aspect_ratio", "symmetry_score", "solidity"):
        out[f"shape_{k}"] = float(f["shape"][k])
    for r in (1, 2):
        for i, v in enumerate(lbp_histogram(image_bgr, mask, radius=r)):
            out[f"texture_r{r}_{i}"] = float(v)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[valid].astype(float)
    for c, name in enumerate(("L", "a", "b")):
        v = lab[:, c]
        out[f"lab_{name}_mean"] = v.mean()
        out[f"lab_{name}_spread"] = v.std()
        out[f"lab_{name}_p10"], out[f"lab_{name}_p90"] = np.percentile(v, [10, 90])

    hs = np.histogram2d(hsv[:, :, 0][valid], hsv[:, :, 1][valid], bins=HS_BINS, range=((0, 180), (40, 256)))[0]
    hs = hs.ravel() / max(hs.sum(), 1)
    out.update({f"hs_{i:02d}": v for i, v in enumerate(hs)})
    return out


def make_logreg(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def predict_variety(saved, image_bgr):
    model, columns = saved["model"], saved["columns"]
    rows = []
    for change in LIGHTING_CHANGES.values():
        img = change(image_bgr)
        f = feature_vector(img, segment_refined(img)) or {}
        rows.append([f.get(c, 0.0) for c in columns])
    probs = model.predict_proba(np.array(rows)).mean(axis=0)
    order = np.argsort(-probs)
    return [(model.classes_[i], float(probs[i])) for i in order]


def make_svm(seed=0):
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    grid = {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]}
    return GridSearchCV(model, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed))
