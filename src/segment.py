"""Find the apple.

    segment(image_bgr) -> mask (255 = apple)

Edge method: blur -> Canny -> close gaps -> convex hull, with GrabCut as a
fallback. Also segment_color (Allyson's method), segment_hybrid and
segment_refined.
"""
import cv2
import numpy as np

# plausible apple size as a fraction of the image
MIN_AREA_FRAC = 0.02
MAX_AREA_FRAC = 0.90


def blur_gaussian(gray, ksize=7):
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def blur_kuwahara(bgr, radius=4):
    """Kuwahara filter: smooths regions but keeps edges sharp."""
    img = bgr.astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = radius + 1
    ksz = (k, k)

    def box(a):
        return cv2.blur(a, ksz, borderType=cv2.BORDER_REFLECT)

    mean = box(gray)
    var = box(gray * gray) - mean * mean

    # shift the box means so each pixel sees its four corner quadrants
    r = radius // 2 + (radius % 2)
    offs = [(r, r), (r, -r), (-r, r), (-r, -r)]

    def shift(a, dy, dx):
        return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

    vars_ = np.stack([shift(var, *o) for o in offs])
    best = np.argmin(vars_, axis=0)

    out = np.empty_like(img)
    for c in range(3):
        m = box(img[:, :, c])
        ms = np.stack([shift(m, *o) for o in offs])
        out[:, :, c] = np.take_along_axis(ms, best[None], axis=0)[0]
    return np.clip(out, 0, 255).astype(np.uint8)


BLURS = {
    "gaussian": lambda bgr: blur_gaussian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
    "kuwahara": lambda bgr: cv2.cvtColor(blur_kuwahara(bgr), cv2.COLOR_BGR2GRAY),
}


def _auto_canny(gray, sigma=0.33):
    med = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * med))
    hi = int(min(255, (1.0 + sigma) * med))
    # keep thresholds usable on dark images
    hi = max(hi, 40)
    lo = max(min(lo, hi // 2), 10)
    return cv2.Canny(gray, lo, hi)


def _largest_hull_mask(edges, shape):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # skip contours touching the border (table or cloth edges)
    h, w = shape[:2]
    m = 3
    inside = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if x > m and y > m and x + cw < w - m and y + ch < h - m:
            inside.append(c)
    if not inside:
        return None
    # rank by length, since a partly found apple edge is an open arc with little area
    arc = lambda c: cv2.arcLength(c, False)
    main = max(inside, key=arc)
    mx, my, mw, mh = cv2.boundingRect(main)
    gap = 0.1 * max(h, w)
    parts = [main]
    for c in inside:
        if c is main or arc(c) < 0.3 * arc(main):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        near = (x < mx + mw + gap and x + cw > mx - gap and
                y < my + mh + gap and y + ch > my - gap)
        if near:
            parts.append(c)
    hull = cv2.convexHull(np.vstack(parts))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
    return mask


def _grabcut_mask(bgr, iters=5):
    """Backup segmentation: GrabCut seeded with a central rectangle."""
    h, w = bgr.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    # fixed seed so results are the same on every run
    cv2.setRNGSeed(0)
    gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    cnt = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [cnt], -1, 255, thickness=cv2.FILLED)
    return out


def _refine_by_saturation(bgr, mask, neutral_sat=40, min_neutral_frac=0.10,
                          min_keep_frac=0.4):
    """Trim gray background (white cloth, counter) that leaked into the mask."""
    if mask is None:
        return mask
    inside = mask > 0
    n = int(inside.sum())
    if n < 100:
        return mask
    sat = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)
    neutral = inside & (sat < neutral_sat)
    if neutral.sum() < min_neutral_frac * n:
        return mask

    keep = (inside & ~neutral).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, k)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    cnt = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [cnt], -1, 255, thickness=cv2.FILLED)
    if np.count_nonzero(out) < min_keep_frac * n:
        return mask
    return out


def _saturation_hull_mask(bgr):
    """Convex hull of the most saturated blob (Otsu on saturation)."""
    sat = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
    _, fg = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = bgr.shape[:2]
    m = 3
    ok = [c for c in contours
          if cv2.boundingRect(c)[0] > m and cv2.boundingRect(c)[1] > m
          and cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] < w - m
          and cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] < h - m]
    if not ok:
        return None
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [cv2.convexHull(max(ok, key=cv2.contourArea))], -1, 255, cv2.FILLED)
    return out


def _complete_with_saturation(bgr, mask, min_iou=0.6):
    """If the saturation outline agrees closely with the edge outline, use the
    hull of both (fills in a missing edge piece). Otherwise leave mask alone."""
    sm = _saturation_hull_mask(bgr)
    if sm is None or not _plausible(sm):
        return mask
    a, b = mask > 0, sm > 0
    if (a & b).sum() / max((a | b).sum(), 1) < min_iou:
        return mask
    pts = np.vstack([cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                     for x in (mask, sm)])
    out = np.zeros_like(mask)
    cv2.drawContours(out, [cv2.convexHull(pts)], -1, 255, cv2.FILLED)
    return out


def _plausible(mask):
    """Reject masks that are too small, too large or a thin arc instead of an apple."""
    if mask is None:
        return False
    frac = float(np.count_nonzero(mask)) / mask.size
    if not (MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC):
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    h, w = mask.shape[:2]
    if bw < 0.15 * w or bh < 0.15 * h:
        return False
    bbox_aspect = bw / max(bh, 1)
    if not (0.50 <= bbox_aspect <= 1.80):
        return False
    return True


def segment(image_bgr, blur="gaussian", max_side=800, return_method=False, refine=True):
    """Return a mask (255 = apple) the same size as image_bgr.

    return_method=True also returns "canny" or "grabcut".
    """
    h0, w0 = image_bgr.shape[:2]
    work = image_bgr
    if max_side and max(h0, w0) > max_side:
        s = max_side / max(h0, w0)
        work = cv2.resize(image_bgr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)

    gray = BLURS[blur](work)
    mask = _largest_hull_mask(_auto_canny(gray), work.shape)
    method = "canny"
    if _plausible(mask):
        mask = _complete_with_saturation(work, mask)
    if not _plausible(mask):
        mask = _grabcut_mask(work)
        method = "grabcut"
    if refine:
        original_mask = mask.copy()
        refined = _refine_by_saturation(work, mask)
        mask = refined if _plausible(refined) else original_mask

    if mask.shape[:2] != (h0, w0):
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (mask, method) if return_method else mask


def segment_color(image_bgr, return_method=False):
    """Allyson's color-mask segmenter (allyson/detect_apple.py), same interface as segment()."""
    from allyson.detect_apple import find_apple, get_apple_mask, get_color_masks, get_hsv

    hsv = get_hsv(image_bgr)
    contour, _ = find_apple(get_apple_mask(hsv, get_color_masks(hsv)))
    mask = np.zeros(image_bgr.shape[:2], np.uint8)
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    method = "color" if contour is not None else "color-none"
    return (mask, method) if return_method else mask


def ellipse_fit(mask):
    """IoU between the mask and the ellipse fitted to it (low = not apple-shaped)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:  # fitEllipse needs at least 5 points
        return 0.0
    ell = np.zeros_like(mask)
    cv2.ellipse(ell, cv2.fitEllipse(cnt), 255, thickness=cv2.FILLED)
    a, b = mask > 0, ell > 0
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


def segment_hybrid(image_bgr, return_method=False):
    """Run the edge and color segmenters and keep the more apple-shaped mask."""
    edge, edge_method = segment(image_bgr, return_method=True)
    color, color_method = segment_color(image_bgr, return_method=True)
    if (color_method != "color-none" and _plausible(color)
            and ellipse_fit(color) > ellipse_fit(edge)):
        mask, method = color, "hybrid-color"
    else:
        mask, method = edge, f"hybrid-{edge_method}"
    return (mask, method) if return_method else mask


def _fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return out


def refine_grabcut(image_bgr, mask, max_side=400, iters=5, grow=0.15, min_frac=0.04):
    """Use a rough mask as the starting guess for GrabCut to fix leaks and missing parts."""
    h0, w0 = image_bgr.shape[:2]
    s = min(1.0, max_side / max(h0, w0))
    img = cv2.resize(image_bgr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else image_bgr
    m = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    h, w = img.shape[:2]

    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    if m.mean() < min_frac:
        x0, y0, x1, y1 = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)
        gc[y0:y1, x0:x1] = cv2.GC_PR_FGD
    else:
        ys, xs = np.nonzero(m)
        dy, dx = int((ys.max() - ys.min()) * grow), int((xs.max() - xs.min()) * grow)
        y0, y1 = max(0, ys.min() - dy), min(h, ys.max() + dy + 1)
        x0, x1 = max(0, xs.min() - dx), min(w, xs.max() + dx + 1)
        gc[y0:y1, x0:x1] = cv2.GC_PR_BGD
        k = max(3, int(0.05 * min(h, w))) | 1
        core = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        gc[m] = cv2.GC_PR_FGD
        gc[core] = cv2.GC_PR_FGD
    if not (gc == cv2.GC_BGD).any() and not (gc == cv2.GC_PR_BGD).any():
        return mask

    cv2.setRNGSeed(0)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    out = _fill_holes(out)
    if np.count_nonzero(out) < 0.01 * out.size:
        return mask
    return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_NEAREST)


def segment_refined(image_bgr, return_method=False):
    """segment_hybrid, then GrabCut refinement (refine_grabcut)."""
    mask, method = segment_hybrid(image_bgr, return_method=True)
    mask = refine_grabcut(image_bgr, mask)
    return (mask, f"{method}+refined") if return_method else mask


SEGMENTERS = {
    "edge": segment,
    "color": segment_color,
    "hybrid": segment_hybrid,
    "refined": segment_refined,
}


def save_overlay(image_bgr, mask, path):
    """Draw the apple outline on the image and save it (for eyeballing failures)."""
    out = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 255), 3)
    cv2.imwrite(str(path), out)
