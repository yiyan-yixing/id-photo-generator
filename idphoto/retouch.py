"""Frequency-separation retouch.

Only the low-frequency layer is smoothed; a controllable share of the high
frequency (pores, skin texture) is kept so the face does not go plastic --
important because school e-photos go through face verification.

Everything here is a skin-only operation, and `retouch` returns the *subject*
RGB with no background baked in. That matters: the caller composites the result
onto white, blue and red, and a result that already had white mixed into its
soft-alpha fringe would leave a pale halo on the coloured backgrounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RetouchParams:
    """Deliberately gentle.

    The output has to survive automated face verification, and an over-whitened
    face both reads as fake and drifts away from the person. The whitening is
    capped so the skin stays inside the natural range: measured on the reference
    photo it moves L by about +4 and leaves b* at ~+14, comfortably inside the
    +12..+25 band for skin. Pushing `white` to 7 drove b* to +12, the edge of
    that band, which is where the face starts to look grey.
    """
    smooth: float = 0.25     # share of the high frequency removed
    spot: float = 0.55       # dark-spot lift
    white: float = 3.5       # L lift on skin
    red: float = 0.08        # a* reduction, as a FRACTION of the pixel's own a*
    yellow: float = 0.12     # b* reduction, as a FRACTION of the pixel's own b*
    undereye: float = 0.20   # under-eye shadow lift, 0 = off


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Feathered 0..1 skin mask: YCbCr colour gate, cleaned and softened."""
    ycc = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = ((Cr >= 133) & (Cr <= 178) & (Cb >= 77) & (Cb <= 130) & (Y >= 60)).astype(np.uint8)

    # keep only the largest connected region (face+neck); drops stray matches
    # on uniform stripes and background
    n, lab, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        skin = (lab == big).astype(np.uint8)

    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return np.clip(cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 12) * 1.6, 0, 1)


def _guided(I: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing of I guided by itself (fast, box-filter based)."""
    k = (2 * radius + 1, 2 * radius + 1)
    mI = cv2.boxFilter(I, -1, k)
    mII = cv2.boxFilter(I * I, -1, k)
    var = mII - mI * mI
    a = var / (var + eps)
    b = mI - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def _under_eye_mask(shape: tuple[int, int], eyes, soft: np.ndarray,
                    eye_specs: list[tuple[float, float, float, float]]) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ue = np.zeros((H, W), np.float32)
    for cx, cy, ew, eh in eye_specs:
        if ew <= 0 or eh <= 0:
            continue
        # sit the ellipse just below the lid: ~0.35 to ~1.75 eye-heights under
        # the eye centre, so the lid itself is not washed out
        ax, ay = 0.85 * ew, 0.72 * eh
        d = ((xx - cx) / ax) ** 2 + ((yy - (cy + 1.02 * eh)) / ay) ** 2
        ue = np.maximum(ue, np.clip(1.0 - d, 0.0, 1.0))
    return cv2.GaussianBlur(np.clip(ue * 2.2, 0, 1) * soft, (0, 0), 12)


def retouch(composite_white: np.ndarray, subject_rgb: np.ndarray, alpha: np.ndarray,
            eye_specs, params: RetouchParams) -> tuple[np.ndarray, dict]:
    """Retouch the subject and return (subject_rgb_retouched, stats).

    `composite_white` is the subject composited onto white -- the working image,
    because the smoothing wants a continuous background. `subject_rgb` is the
    untouched pixels, which is what we blend back toward outside the skin.
    """
    soft = skin_mask(composite_white)

    lab = cv2.cvtColor(composite_white.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]

    # 1. frequency separation: attenuate the high frequency, keep the edges
    base = _guided(L, 18, 1e-3 * 255 * 255)
    L_new = base + (L - base) * (1.0 - params.smooth)

    # 2. spot lift -- reference is a wide blur, so broad shading barely registers
    ref = cv2.GaussianBlur(L_new, (0, 0), 45)
    L_new = L_new + np.clip(ref - L_new, 0, 22) * params.spot

    # 3. under-eye shadow lift
    if params.undereye > 0 and eye_specs:
        ue = _under_eye_mask(L.shape, None, soft, eye_specs)
        # Reference = local bright envelope of *skin only*: a wide max-filter
        # over the skin gives the surrounding healthy skin tone, so the deficit
        # measures how much darker the bag is than the cheek beside it. Non-skin
        # is zeroed first and dilate() takes a max, so the background -- white
        # here, any colour later -- cannot bleed into the reference.
        L_skin = np.where(soft > 0.02, L_new, 0).astype(np.float32)
        env = cv2.GaussianBlur(
            cv2.dilate(L_skin, cv2.getStructuringElement(cv2.MORPH_RECT, (151, 151))),
            (0, 0), 40)
        L_new = L_new + np.clip(env - L_new, 0, 60) * params.undereye * ue

    # 4. whitening
    m = soft
    lab[..., 0] = np.clip(L_new + params.white * m, 0, 255)
    # Chroma is scaled, not shifted. Subtracting a constant removed the same
    # 1.0 a* / 2.0 b* everywhere, which on bright skin is a 10% change but on
    # shadowed skin (b* around 6) strips a third of its warmth -- and those
    # areas then read as a grey-green cast around the eyesocket, brow and
    # hairline. Scaling keeps highlights' saturation and barely touches shadows.
    a_ch = lab[..., 1] - 128.0
    b_ch = lab[..., 2] - 128.0
    lab[..., 1] = np.clip(a_ch * (1.0 - params.red * m) + 128.0, 0, 255)
    lab[..., 2] = np.clip(b_ch * (1.0 - params.yellow * m) + 128.0, 0, 255)

    edited = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    worked = np.clip(edited * m[..., None] +
                     composite_white.astype(np.float32) * (1 - m[..., None]), 0, 255).astype(np.uint8)

    # Blend back into the untouched subject. The weight is `soft * alpha` rather
    # than `soft` alone: it goes to zero both outside the skin and wherever the
    # matte is semi-transparent, so the white working image can never leak into
    # the hair fringe.
    w = (soft * alpha)[..., None]
    out = subject_rgb.astype(np.float32) * (1 - w) + worked.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8), {
        "skin_pct": float((soft > 0.5).mean() * 100),
    }
