"""Compare the variety methods with grouped cross-validation, plus a lighting test
and a nested estimate for the final model. Writes results/species_*.csv and
results/figures/.

    python compare_species.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.classifier import FINAL_PREFIXES, feature_vector, make_logreg, make_svm
from src.lighting import LIGHTING_CHANGES
from src.segment import refine_grabcut, segment_hybrid
from src.species import VARIETIES, apple_features, classify_species

REPEATS = 5
FOLDS = 5
CONDITIONS = LIGHTING_CHANGES
USE_CNN = Path("models/mobilenetv2-12.onnx").exists()

CST = ("color_", "shape_", "texture_")
ALL = CST + ("lab_", "hs_")
FINAL = "Final: logistic regression, refined outlines + lighting copies + averaging"

METHODS = {
    "SVM: color": dict(p=("color_",), gw=False, seg="hybrid", copies=False, avg=False, make=make_svm),
    "SVM: color + shape": dict(p=("color_", "shape_"), gw=False, seg="hybrid", copies=False, avg=False, make=make_svm),
    "SVM: color + shape + texture": dict(p=CST, gw=False, seg="hybrid", copies=False, avg=False, make=make_svm),
    "SVM: + gray world": dict(p=CST, gw=True, seg="hybrid", copies=False, avg=False, make=make_svm),
    "SVM: + lighting copies": dict(p=CST, gw=False, seg="hybrid", copies=True, avg=False, make=make_svm),
    "SVM: refined outlines": dict(p=CST, gw=False, seg="refined", copies=False, avg=False, make=make_svm),
    "SVM: refined + Lab + hue/sat + lighting copies": dict(p=ALL, gw=False, seg="refined", copies=True, avg=False, make=make_svm),
    FINAL: dict(p=FINAL_PREFIXES, gw=False, seg="refined", copies=True, avg=True, make=make_logreg),
}
if USE_CNN:
    METHODS["CNN features (MobileNetV2) + lighting copies, SVM"] = dict(
        p=("cnn_",), gw=False, seg="refined", copies=True, avg=False, make=make_svm)
RULES = {"Rules": "hybrid"}

NESTED_CANDIDATES = {
    "logreg C=0.1, color+shape+texture": (CST, lambda: make_logreg(0.1)),
    "logreg C=1, color+shape+texture": (CST, lambda: make_logreg(1.0)),
    "SVM C=1, color+shape+texture": (CST, lambda: make_pipeline(
        StandardScaler(), CalibratedClassifierCV(SVC(C=1, class_weight="balanced"), ensemble=False))),
    "logreg C=0.1, + Lab + hue/sat": (ALL, lambda: make_logreg(0.1)),
    "random forest, + Lab + hue/sat": (ALL, lambda: RandomForestClassifier(
        300, class_weight="balanced", random_state=0, n_jobs=-1)),
}

SURFACE, INK, INK_2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def wilson(correct, n, z=1.96):
    """95% interval for an accuracy measured on n photos."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def measure_all(labels):
    """Features and rule guesses for every photo under every lighting condition."""
    if USE_CNN:
        from src.cnn import cnn_features
    feats, rules = {}, {}
    for i, r in labels.iterrows():
        original = cv2.imread(f"data/species/{r.filename}")
        for cond, change in CONDITIONS.items():
            img = change(original)
            hybrid = segment_hybrid(img)
            refined = refine_grabcut(img, hybrid)
            for gw in (False, True):
                feats[(i, cond, "hybrid", gw)] = feature_vector(img, hybrid, lighting=gw) or {}
            f = feature_vector(img, refined) or {}
            if USE_CNN:
                f.update({f"cnn_{k:04d}": v for k, v in enumerate(cnn_features(img, refined))})
            feats[(i, cond, "refined", False)] = f
            rules[(i, cond, "hybrid")] = classify_species(apple_features(img, hybrid))[0]
        print(f"measured {r.filename}")
    return feats, rules


def to_matrix(feats, idx, cond, seg, gw, prefixes, columns=None):
    rows = [{k: v for k, v in feats[(i, cond, seg, gw)].items() if k.startswith(prefixes)} for i in idx]
    df = pd.DataFrame(rows, index=list(idx))
    if columns is not None:
        df = df.reindex(columns=columns)
    return df.fillna(0.0)


def train(feats, idx, y, m):
    X = to_matrix(feats, idx, "normal", m["seg"], m["gw"], m["p"])
    yt = list(y)
    if m["copies"]:
        extra = [to_matrix(feats, idx, c, m["seg"], m["gw"], m["p"], X.columns) for c in CONDITIONS if c != "normal"]
        X = pd.concat([X] + extra)
        yt = yt * len(CONDITIONS)
    return m["make"]().fit(X.values, yt), X.columns


def predict(model, columns, feats, idx, cond, m):
    """Best guess and the top two, for one lighting condition."""
    if m["avg"]:
        # average over the lighting versions only for the normal photo
        conds = list(CONDITIONS) if cond == "normal" else [cond]
        probs = np.mean([model.predict_proba(to_matrix(feats, idx, c, m["seg"], m["gw"], m["p"], columns).values)
                         for c in conds], axis=0)
        order = np.argsort(-probs, axis=1)
        return model.classes_[order[:, 0]], [set(model.classes_[o[:2]]) for o in order]
    guess = model.predict(to_matrix(feats, idx, cond, m["seg"], m["gw"], m["p"], columns).values)
    return guess, [None] * len(guess)


def cross_validate(labels, feats):
    """One row per (repeat, method, condition, photo) with the guess."""
    y = labels.variety.values
    groups = labels.group.values
    out = []
    for rep in range(REPEATS):
        cv = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=rep)
        for fold, (train_i, test_i) in enumerate(cv.split(labels, y, groups)):
            tr_idx, te_idx = labels.index[train_i], labels.index[test_i]
            for name, m in METHODS.items():
                model, columns = train(feats, tr_idx, y[train_i], m)
                for cond in CONDITIONS:
                    guess, top2 = predict(model, columns, feats, te_idx, cond, m)
                    out += [{"repeat": rep, "fold": fold, "method": name, "condition": cond, "row": i, "guess": g,
                             "in_top2": (labels.variety[i] in t) if t is not None else np.nan}
                            for i, g, t in zip(te_idx, guess, top2)]
            print(f"repeat {rep} fold {fold} done")
    return pd.DataFrame(out)


def summarise(labels, preds, rules):
    held = labels.split != "tune"
    truth = labels.variety
    rows, robust = [], []
    for name in list(RULES) + list(METHODS):
        for cond in CONDITIONS:
            top2 = np.nan
            if name in RULES:
                guess = pd.Series({i: rules[(i, cond, RULES[name])] for i in labels.index})
                accs_held, accs_all = [(guess[held] == truth[held]).mean()], [np.nan]
            else:
                p = preds[(preds.method == name) & (preds.condition == cond)]
                accs_held, accs_all = [], []
                for _, rep in p.groupby("repeat"):
                    g = rep.set_index("row").guess.reindex(labels.index)
                    accs_held.append((g[held] == truth[held]).mean())
                    accs_all.append((g == truth).mean())
                if p.in_top2.notna().any():
                    top2 = p.in_top2.astype(float).mean()
            n_held = int(held.sum())
            lo, hi = wilson(np.mean(accs_held) * n_held, n_held)
            rec = {"method": name, "condition": cond, "accuracy_heldout": np.mean(accs_held),
                   "ci_low": lo, "ci_high": hi, "spread_over_repeats": np.std(accs_held), "n_heldout": n_held,
                   "accuracy_all_cv": np.nan if np.all(np.isnan(accs_all)) else np.nanmean(accs_all),
                   "top2_all_cv": top2, "n_all": len(labels)}
            robust.append(rec)
            if cond == "normal":
                rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(robust)


def nested_estimate(labels, feats, repeats=3):
    """Pick the model inside each training split, score it on the outer test split."""
    y, g = labels.variety.values, labels.group.values

    def fit_predict(p, make, tr, te):
        X = np.vstack([to_matrix(feats, tr, c, "refined", False, p).values for c in CONDITIONS])
        model = make().fit(X, np.tile(labels.variety[tr].values, len(CONDITIONS)))
        cols = to_matrix(feats, tr, "normal", "refined", False, p).columns
        probs = np.mean([model.predict_proba(to_matrix(feats, te, c, "refined", False, p, cols).values)
                         for c in CONDITIONS], axis=0)
        return model.classes_[probs.argmax(1)]

    rows = []
    for rep in range(repeats):
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=100 + rep).split(labels, y, g)):
            tr_idx, te_idx = labels.index[tr], labels.index[te]
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=rep)
            scores = {}
            for name, (p, make) in NESTED_CANDIDATES.items():
                scores[name] = np.mean([np.mean(fit_predict(p, make, tr_idx[a], tr_idx[b]) == y[tr][b])
                                        for a, b in inner.split(tr_idx, y[tr], g[tr])])
            pick = max(scores, key=scores.get)
            correct = fit_predict(*NESTED_CANDIDATES[pick], tr_idx, te_idx) == y[te]
            rows.append({"repeat": rep, "fold": fold, "picked": pick, "n_test": len(te), "correct": int(correct.sum())})
    df = pd.DataFrame(rows)
    sums = df.groupby("repeat")[["correct", "n_test"]].sum()
    per_rep = sums.correct / sums.n_test
    return df, per_rep.mean(), per_rep.std()


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def short(name):
    return "Final model (logistic regression)" if name == FINAL else name


def plot_accuracy(summary, path):
    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(s) + 1.0), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(s))
    ax.errorbar(s.accuracy_heldout, y, xerr=[s.accuracy_heldout - s.ci_low, s.ci_high - s.accuracy_heldout],
                fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, capsize=0, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    for yy, a, hi in zip(y, s.accuracy_heldout, s.ci_high):
        ax.text(hi + 0.015, yy, f"{a:.0%}", va="center", fontsize=9, color=INK)
    ax.axvline(1 / len(VARIETIES), color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
    ax.text(1 / len(VARIETIES) + 0.01, len(s) - 0.45, "guessing (17%)", fontsize=8, color=MUTED, va="bottom")
    ax.set_ylim(-0.6, len(s) - 0.1)
    ax.set_yticks(y, [short(n) for n in s.method], color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Variety accuracy on {int(s.n_heldout.iloc[0])} held-out photos (95% interval)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_confusions(labels, preds, rules, path):
    held = labels[labels.split != "tune"]
    names = list(VARIETIES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    p = preds[(preds.method == FINAL) & (preds.condition == "normal") & (preds.repeat == 0)].set_index("row").guess
    panels = []
    for title, guess in (("Rules", {i: rules[(i, "normal", "hybrid")] for i in held.index}),
                         ("Final model", {i: p[i] for i in held.index})):
        m = np.zeros((len(names), len(names)), int)
        for i, r in held.iterrows():
            if guess[i] in names:
                m[names.index(r.variety), names.index(guess[i])] += 1
        panels.append((title, m))
    top = max(m.max() for _, m in panels)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    for ax, (title, m) in zip(axes, panels):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=top)
        for a in range(len(names)):
            for b in range(len(names)):
                if m[a, b]:
                    ax.text(b, a, m[a, b], ha="center", va="center", fontsize=9,
                            color=SURFACE if m[a, b] >= top * 0.55 else INK)
        labels_short = [VARIETIES[n].replace(" Delicious", " Del.") for n in names]
        ax.set_xticks(range(len(names)), labels_short, rotation=35, ha="right", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(names)), labels_short, fontsize=8, color=INK_2)
        ax.set_xlabel("guess", color=INK_2, fontsize=9)
        ax.set_ylabel("true variety", color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_robustness(robust, path):
    show = ["Rules", "SVM: color + shape + texture", "SVM: + gray world", FINAL]
    names = ["Rules", "SVM", "SVM + gray world", "Final model"]
    conds = list(CONDITIONS)
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    width = 0.2
    for k, (m, lab) in enumerate(zip(show, names)):
        vals = [robust[(robust.method == m) & (robust.condition == c)].accuracy_heldout.iloc[0] for c in conds]
        x = np.arange(len(conds)) + (k - 1.5) * (width + 0.01)
        ax.bar(x, vals, width, color=SERIES[k], label=lab, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(np.arange(len(conds)), conds, color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, labelcolor=INK)
    ax.set_title("Held-out accuracy when the lighting changes", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_failures(labels, preds, path, limit=8):
    p = preds[(preds.method == FINAL) & (preds.condition == "normal") & (preds.repeat == 0)].set_index("row").guess
    held = labels[labels.split != "tune"]
    wrong = [(i, r) for i, r in held.iterrows() if p[i] != r.variety][:limit]
    if not wrong:
        return
    tiles = []
    for i, r in wrong:
        img = cv2.imread(f"data/species/{r.filename}")
        mask = refine_grabcut(img, segment_hybrid(img))
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 0, 255), max(2, img.shape[0] // 120))
        k = 220 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=k, fy=k)
        tile = np.full((270, 220, 3), 255, np.uint8)
        tile[:img.shape[0], :img.shape[1]] = img
        cv2.putText(tile, f"true: {VARIETIES[r.variety]}", (4, 240), 0, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, f"guess: {VARIETIES.get(p[i], p[i])}", (4, 260), 0, 0.42, (0, 0, 180), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.full((270, 220, 3), 255, np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)]))


def main():
    labels = pd.read_csv("data/species_labels.csv")
    feats, rules = measure_all(labels)
    preds = cross_validate(labels, feats)
    summary, robust = summarise(labels, preds, rules)
    nested, nested_mean, nested_sd = nested_estimate(labels, feats)

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    summary.to_csv("results/species_comparison.csv", index=False)
    robust.to_csv("results/species_robustness.csv", index=False)
    nested.to_csv("results/species_nested.csv", index=False)
    preds.merge(labels[["filename", "variety", "split"]], left_on="row", right_index=True) \
         .drop(columns="row").to_csv("results/species_cv_predictions.csv", index=False)

    plot_accuracy(summary, "results/figures/species_accuracy.png")
    plot_confusions(labels, preds, rules, "results/figures/species_confusion.png")
    plot_robustness(robust, "results/figures/species_robustness.png")
    plot_failures(labels, preds, "results/figures/species_failures.png")

    pd.set_option("display.width", 160)
    print(summary[["method", "accuracy_heldout", "ci_low", "ci_high", "spread_over_repeats",
                   "accuracy_all_cv", "top2_all_cv"]].round(3).to_string(index=False))
    print()
    print(robust.pivot(index="method", columns="condition", values="accuracy_heldout")[list(CONDITIONS)].round(3).to_string())
    print(f"\nnested estimate (model picked without the test photos): {nested_mean:.3f} +/- {nested_sd:.3f}")
    print(nested.picked.value_counts().to_string())


if __name__ == "__main__":
    main()
