"""Learned alpha matting, replacing the heuristic matte refinement.

The heuristic that used to live here inferred coverage from a two-tone relation
(`a = (C - B) / (F - B)`) on top of the Vision segmentation. That works on the
silhouette boundary, but it is a *local* classifier and it is deliberately locked
out inside the solid region -- so when Vision mislabels a patch of background as
subject, nothing downstream can ever correct it. Measured on the reference photo:
85,442 pixels inside the silhouette were background, the largest single blob
28,896 px. That survived every parameter I tried, because the fault is in the
segmentation, not in the edge.

RMBG-1.4 is trained for exactly this and produces a proper matte directly: on the
same photo the interior blob is simply absent, and inference costs about a
second.

The model file is fetched from hf-mirror (HuggingFace proper is unreachable
here). If it is missing, the caller falls back to the Vision mask rather than
failing.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "vendor" / "models"
MODEL_SIZE = 1024

# Both take a 1024x1024 float image and return a 1024x1024 map, but they do not
# agree on how to read that map or on what to feed in:
#   rmbg14   : input (x - 0.5);            output is already a probability
#   birefnet : input ImageNet mean/std;    output is a big raw score -> sigmoid
# Feeding one the other's convention produces a confident-looking but wrong
# matte (BiRefNet with min-max normalisation segments only the face), so the
# choice is part of the backend, not a detail.
BACKENDS = {
    "birefnet": ("birefnet.onnx", "imagenet", "sigmoid"),
    "rmbg":     ("rmbg14.onnx",   "half",     "minmax"),
}
DEFAULT_BACKEND = "birefnet"

_sessions: dict = {}
_lock = threading.Lock()


def available(backend: str = DEFAULT_BACKEND) -> bool:
    return (MODELS_DIR / BACKENDS[backend][0]).is_file()


def _load(backend: str):
    if backend not in _sessions:
        with _lock:
            if backend not in _sessions:
                import onnxruntime as ort
                _sessions[backend] = ort.InferenceSession(
                    str(MODELS_DIR / BACKENDS[backend][0]),
                    providers=["CPUExecutionProvider"])
    return _sessions[backend]


def _predict_tile(bgr: np.ndarray, backend: str) -> np.ndarray:
    """Run the network on one crop, returning alpha at that crop's resolution."""
    sess = _load(backend)
    h, w = bgr.shape[:2]
    x = cv2.resize(bgr, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LANCZOS4)
    x = x.astype(np.float32) / 255.0
    _, norm, post = BACKENDS[backend]
    if norm == "imagenet":
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32)
    else:
        x = x - 0.5

    out = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    if post == "sigmoid":
        out = 1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))
    else:
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LANCZOS4)


def predict(rgb: np.ndarray, tiles: int = 1, overlap: float = 0.25,
            backend: str = DEFAULT_BACKEND) -> np.ndarray:
    """Soft alpha in [0, 1] at the input's resolution.

    The network takes a fixed 1024x1024 square and is trained on squashed input,
    so on a full frame the aspect ratio is deliberately not preserved -- padding
    would put the subject outside the distribution it learned.

    That fixed input is also the ceiling on detail: a 12MP photo is squeezed to
    1024 on its long edge, so a strand a few pixels wide does not survive, and the
    resulting matte cannot resolve individual hairs however good the model is.
    Running `tiles` x `tiles` overlapping crops instead gives each region its own
    1024 window, which is the same as matting at 2-3x the resolution. The price
    is one forward pass per tile, roughly a second each.

    Tiles are blended with a feathered weight so the seams fall where both
    neighbours are confident.

    More tiles is not monotonically better. At 3x3 each crop holds a smaller
    share of the subject, which drifts outside what the model saw in training,
    and the fringe comes out washed-out and grey -- measured and confirmed by
    the vision check, which preferred 2x2. 2x2 doubles the effective matting
    resolution, which is where the visible gain is.
    """
    if tiles <= 1:
        return _sharpen(rgb, np.clip(_predict_tile(rgb, backend), 0, 1))

    h, w = rgb.shape[:2]
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    step = 1.0 / tiles
    for ty in range(tiles):
        for tx in range(tiles):
            # widen each tile by the overlap on every side that has a neighbour,
            # so adjacent windows share a band and can be cross-faded
            x0 = max(0, int((tx * step - overlap) * w))
            x1 = min(w, int(((tx + 1) * step + overlap) * w))
            y0 = max(0, int((ty * step - overlap) * h))
            y1 = min(h, int(((ty + 1) * step + overlap) * h))
            tile = _predict_tile(rgb[y0:y1, x0:x1], backend)

            # feather to zero at the tile's outer edges: only the interior of each
            # tile is trusted, which is why the tiles overlap in the first place
            win = np.outer(_ramp(y1 - y0), _ramp(x1 - x0)).astype(np.float32)
            acc[y0:y1, x0:x1] += tile * win
            wsum[y0:y1, x0:x1] += win

    alpha = np.clip(acc / np.maximum(wsum, 1e-6), 0, 1)
    return _sharpen(rgb, alpha)


def _ramp(n: int, edge: float = 0.25) -> np.ndarray:
    """Trapezoid going to zero at both ends, flat in the middle."""
    r = np.ones(n, np.float32)
    k = max(1, int(n * edge))
    r[:k] = np.linspace(0, 1, k, dtype=np.float32)
    r[-k:] = np.linspace(1, 0, k, dtype=np.float32)
    return r


def _sharpen(rgb: np.ndarray, alpha: np.ndarray,
             radius: int = 8, eps: float = 1e-4, contrast: float = 1.6) -> np.ndarray:
    """Pull the matte onto the image's own edges.

    The model runs at a fixed 1024x1024, so upscaled to a 3024-wide photo its
    boundary is inherently soft -- fine strands blur into the background. A
    guided filter with the image as guide transfers the real edge structure back
    into the matte, then a gentle contrast stretch re-tightens the transition.

    Unlike the earlier two-tone heuristic this makes no assumption about what the
    foreground and background colours are, so it cannot invent the interior
    false-positives that motivated switching to a learned matte in the first
    place. Compared against the raw model output, this was judged clearly better
    on edge sharpness, halo suppression and strand retention.
    """
    guide = cv2.GaussianBlur(
        cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0,
        (0, 0), 1.5)
    k = (2 * radius + 1, 2 * radius + 1)
    mI = cv2.boxFilter(guide, -1, k)
    mP = cv2.boxFilter(alpha, -1, k)
    cov = cv2.boxFilter(guide * alpha, -1, k) - mI * mP
    var = cv2.boxFilter(guide * guide, -1, k) - mI * mI
    a = cov / (var + eps)
    b = mP - a * mI
    refined = cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)
    return np.clip((refined - 0.5) * contrast + 0.5, 0, 1)
