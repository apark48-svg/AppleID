"""Size and position of the apple from cv2.fitEllipse (in pixels).

    apple_size(mask) -> dict | None
"""
import cv2


def apple_size(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:  # fitEllipse needs at least 5 points
        return None
    (cx, cy), (d1, d2), _ = cv2.fitEllipse(cnt)
    width, height = sorted((d1, d2))
    x, y, bw, bh = cv2.boundingRect(cnt)
    return {
        "width_px": float(width),
        "height_px": float(height),
        "aspect": float(height / width) if width > 0 else None,
        "area_px": float(cv2.contourArea(cnt)),
        "center_xy": (float(cx), float(cy)),
        "bbox_xywh": (int(x), int(y), int(bw), int(bh)),
    }
