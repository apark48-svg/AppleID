"""Draw the pipeline's answer on the photo.

    annotate(image_bgr, mask, label, shares, size) -> image

Draws the apple outline, a bounding box, the center point, and a text banner
with the color label and shares (e.g. "RED  r97 y3 g0").
"""
import cv2


def annotate(image_bgr, mask, label, shares, size):
    out = image_bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(h, w) / 600.0)
    thick = max(1, int(round(2 * scale)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 0, 255), thick + 1)

    if size is not None:
        x, y, bw, bh = size["bbox_xywh"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (255, 255, 0), thick)
        cx, cy = (int(round(v)) for v in size["center_xy"])
        r = max(4, int(6 * scale))
        cv2.circle(out, (cx, cy), r, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), r, (0, 0, 0), 1)

    text = label.upper()
    sub = "  ".join(f"{k[0]}{round(v * 100)}" for k, v in shares.items())
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, f, 1.0 * scale, thick + 1)
    (sw, sh), _ = cv2.getTextSize(sub, f, 0.5 * scale, thick)
    cv2.rectangle(out, (0, 0), (max(tw, sw) + 16, th + sh + 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, th + 8), f, 1.0 * scale, (255, 255, 255), thick + 1, cv2.LINE_AA)
    cv2.putText(out, sub, (8, th + sh + 16), f, 0.5 * scale, (200, 200, 200), thick, cv2.LINE_AA)
    return out
