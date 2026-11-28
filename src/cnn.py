"""Pretrained MobileNetV2 features (used only for comparison in compare_species.py).

Needs models/mobilenetv2-12.onnx from the ONNX model zoo, see README.
"""
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path("models/mobilenetv2-12.onnx")
FEATURE_LAYER = "onnx_node!GlobalAveragePool_97"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224

_net = None


def _load():
    global _net
    if _net is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} is missing, see README for the download command")
        _net = cv2.dnn.readNetFromONNX(str(MODEL_PATH), engine=cv2.dnn.ENGINE_CLASSIC)
    return _net


def _square_crop(image_bgr, mask, margin=0.1):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return image_bgr
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) * (0.5 + margin)
    y0, y1, x0, x1 = int(cy - half), int(cy + half), int(cx - half), int(cx + half)
    h, w = image_bgr.shape[:2]
    pad = max(0, -y0, -x0, y1 - h, x1 - w)
    if pad:
        image_bgr = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        y0, y1, x0, x1 = y0 + pad, y1 + pad, x0 + pad, x1 + pad
    return image_bgr[y0:y1, x0:x1]


def cnn_features(image_bgr, mask=None):
    img = image_bgr if mask is None else _square_crop(image_bgr, mask)
    x = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    net = _load()
    net.setInput(x)
    return net.forward(FEATURE_LAYER).ravel()
