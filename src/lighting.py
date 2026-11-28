"""Gray-world color correction and the lighting changes used for testing/training."""
import numpy as np


def gray_world(image_bgr, mask=None, min_pixels=500, max_gain=1.6):
    img = image_bgr.astype(np.float32)
    region = np.ones(img.shape[:2], bool) if mask is None else mask == 0
    if np.count_nonzero(region) < min_pixels:
        region = np.ones(img.shape[:2], bool)

    means = img[region].mean(axis=0)
    gains = means.mean() / np.maximum(means, 1.0)
    # very dark or single-colored backgrounds give extreme gains
    gains = np.clip(gains, 1.0 / max_gain, max_gain)
    return np.clip(img * gains, 0, 255).astype(np.uint8)


def _scale(image_bgr, gains_bgr):
    return np.clip(image_bgr.astype(np.float32) * np.array(gains_bgr, np.float32), 0, 255).astype(np.uint8)


LIGHTING_CHANGES = {
    "normal": lambda im: im,
    "darker": lambda im: _scale(im, (0.55, 0.55, 0.55)),
    "brighter": lambda im: _scale(im, (1.35, 1.35, 1.35)),
    "warmer": lambda im: _scale(im, (0.8, 1.0, 1.15)),
    "cooler": lambda im: _scale(im, (1.15, 1.0, 0.8)),
}
