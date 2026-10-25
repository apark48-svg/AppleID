"""Step 3: find the apple.

Public interface (agreed with the team):
    segment(image_bgr) -> mask   # uint8, 255 = apple, 0 = background

Pipeline: blur -> Canny -> close gaps -> largest contour -> convex hull -> mask.
If that gives an implausible result, fall back to GrabCut.

To try a different blur (e.g. Kuwahara), pass blur="kuwahara" or add a new
entry to BLURS below. Nothing else needs to change.
"""
import cv2
import numpy as np

# Plausible apple size as a fraction of the image area. Outside this range the
# edge-based result is treated as a failure and GrabCut is used instead.
MIN_AREA_FRAC = 0.02
MAX_AREA_FRAC = 0.90


# ---------------------------------------------------------------- blurs ----
def blur_gaussian(gray, ksize=7):
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def blur_kuwahara(bgr, radius=4):
    """Kuwahara filter: smooths regions but keeps edges sharp.

    For each pixel, look at four overlapping (radius+1)x(radius+1) quadrants
    around it and output the mean of the quadrant with the lowest variance.
    Simple numpy version; slow on full-size photos, so segment() downscales
    first.
    """
    img = bgr.astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = radius + 1
    ksz = (k, k)

    def box(a):
        return cv2.blur(a, ksz, borderType=cv2.BORDER_REFLECT)

    mean = box(gray)
    var = box(gray * gray) - mean * mean

    # box() is centered on each pixel; shift it so each pixel sees the four
    # quadrants that have the pixel at a corner.
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


# Each entry: BGR image -> blurred grayscale image used for edge detection.
BLURS = {
    "gaussian": lambda bgr: blur_gaussian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
    "kuwahara": lambda bgr: cv2.cvtColor(blur_kuwahara(bgr), cv2.COLOR_BGR2GRAY),
}


# ------------------------------------------------------------ helpers ------
def _auto_canny(gray, sigma=0.33):
    med = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * med))
    hi = int(min(255, (1.0 + sigma) * med))
    # Dark images give a tiny median; keep thresholds from collapsing to ~0.
    hi = max(hi, 40)
    lo = max(min(lo, hi // 2), 10)
    return cv2.Canny(gray, lo, hi)


def _largest_hull_mask(edges, shape):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # An apple sits inside the frame. Contours that touch the image border are
    # usually background (countertop edge, table edge, cloth), so skip them.
    h, w = shape[:2]
    m = 3
    inside = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if x > m and y > m and x + cw < w - m and y + ch < h - m:
            inside.append(c)
    if not inside:
        return None
    # Rank by edge length, not enclosed area: an apple edge that Canny finds
    # only partly (e.g. the lower half fading into a white cloth) is an open
    # arc with almost no area but a long length.
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
            parts.append(c)  # other long edge pieces next to the main one
    hull = cv2.convexHull(np.vstack(parts))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
    return mask


def _grabcut_mask(bgr, iters=5):
    """Backup segmentation: GrabCut seeded with a central rectangle."""
    h, w = bgr.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    # GrabCut uses random numbers internally. Without a fixed seed the result
    # for one image depends on which images were processed before it, which
    # makes results differ between scripts and between teammates' machines.
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
    """Trim near-gray background (white cloth, gray counter) leaked into the mask.

    Only pixels that are essentially colorless (saturation < neutral_sat) are
    removed, so paler or shaded parts of the apple itself are never cut. The
    trim only runs when at least min_neutral_frac of the mask is colorless,
    which is what a leak looks like; a normal apple mask is returned unchanged.
    """
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
    """Colorful-region outline: Otsu on saturation, largest blob, convex hull.
    Helps where an apple's edge fades into a pale background or shadow but the
    apple is still clearly more colorful than its surroundings."""
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
    """Reject masks that are large enough by area but only capture an edge arc.

    Area alone is not sufficient: a thickened Canny arc along the bottom of an
    apple can occupy >2% of the image and incorrectly pass the old check.
    Apples should have a reasonably two-dimensional bounding box.
    """
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
    # Reject thin strips/arcs and tiny detections. These limits are relative to
    # the image, so they work across different input resolutions.
    if bw < 0.15 * w or bh < 0.15 * h:
        return False
    bbox_aspect = bw / max(bh, 1)
    if not (0.50 <= bbox_aspect <= 1.80):
        return False
    return True


# --------------------------------------------------------------- public ----
def segment(image_bgr, blur="gaussian", max_side=800, return_method=False, refine=True):
    """Return a binary mask (255 = apple) the same size as image_bgr.

    max_side: image is downscaled for speed, then the mask is resized back.
              Set to None to disable.
    return_method: if True, return (mask, "canny" | "grabcut") so you can see
                   how often the backup was needed.
    refine: if True, trim neutral (low-saturation) background that leaked into
            the mask. Set False to compare with the plain result.
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
        # Saturation trimming can occasionally leave only a thin colored strip.
        # Keep it only if it still looks like a whole fruit mask.
        mask = refined if _plausible(refined) else original_mask

    if mask.shape[:2] != (h0, w0):
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (mask, method) if return_method else mask


def segment_color(image_bgr, return_method=False):
    """Allyson's segmenter (allyson/detect_apple.py), same interface as segment().

    Keeps pixels that are both saturated (Otsu) and inside an apple hue range,
    then picks the largest roughly round blob. Tight on vivid apples over gray
    or textured backgrounds; can miss deeply shaded parts of an apple.
    Returns an empty mask if no apple-like blob is found.
    """
    from allyson.detect_apple import find_apple, get_apple_mask, get_color_masks, get_hsv

    hsv = get_hsv(image_bgr)
    contour, _ = find_apple(get_apple_mask(hsv, get_color_masks(hsv)))
    mask = np.zeros(image_bgr.shape[:2], np.uint8)
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    method = "color" if contour is not None else "color-none"
    return (mask, method) if return_method else mask


# Each entry: segmenter with the segment() interface. Add new methods here.
SEGMENTERS = {
    "edge": segment,
    "color": segment_color,
}


def save_overlay(image_bgr, mask, path):
    """Draw the apple outline on the image and save it (for eyeballing failures)."""
    out = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 255), 3)
    cv2.imwrite(str(path), out)
