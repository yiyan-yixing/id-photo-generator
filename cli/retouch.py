#!/usr/bin/env python3
"""Frequency-separation retouch for a composited ID photo (white background).

Only the low-frequency layer is smoothed; a controllable share of the high
frequency (pores, skin texture) is kept so the face does not go plastic --
important because school e-photos go through face verification.

Stages
  1. skin mask   : YCbCr colour gate, cleaned up and feathered
  2. smoothing   : guided-filter base + attenuated detail (pores and small
                   blemishes fade, real edges survive)
  3. spot lift   : compact dark features (moles, spots) lifted toward the
                   surrounding skin tone, broad shading left alone
  4. whitening   : L up, a* and b* down inside the mask, in LAB
  5. mole removal: compact dark blobs inpainted from the surrounding skin

The eye regions that drive the under-eye lift and the blemish keep-out are
detected by Vision rather than typed in: they are pixel coordinates, so any
hand-transcribed value is invalid the moment the input is resized or re-cropped.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageOps

import _vision

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("out")
p.add_argument("--smooth", type=float, default=0.45, help="detail removed, 0-1")
p.add_argument("--spot", type=float, default=0.55, help="dark-spot lift, 0-1")
p.add_argument("--white", type=float, default=2.5, help="L lift on skin")
p.add_argument("--red", type=float, default=0.8, help="a* reduction (redness)")
p.add_argument("--yellow", type=float, default=1.5, help="b* reduction (sallowness)")
p.add_argument("--keep-texture", type=float, default=1.0,
               help="scale on the retained high frequency; <1 = softer")
p.add_argument("--undereye", type=float, default=0.0,
               help="under-eye shadow lift, 0-1 (0 = off)")
p.add_argument("--moles", type=float, default=0.0,
               help="remove compact dark spots/moles inside skin, 0-1 (0 = keep)")
p.add_argument("--eye", action="append", default=[],
               help="override an eye region as cx,cy,w,h in this image's pixels; "
                    "repeat per eye. Detected by Vision when omitted.")
p.add_argument("--dump-mask", default=None)
args = p.parse_args()

# PIL rather than cv2.imread: the latter ignores EXIF orientation entirely, so
# a JPEG straight off a phone would be processed sideways.
_img = ImageOps.exif_transpose(Image.open(args.src)) or Image.open(args.src)
bgr = cv2.cvtColor(np.array(_img.convert("RGB")), cv2.COLOR_RGB2BGR)
H, W = bgr.shape[:2]
print(f"working at {W}x{H}")

# --- eye regions: ask Vision unless the caller pinned them -----------------
if args.eye:
    eye_specs = [tuple(float(v) for v in s.split(",")) for s in args.eye]
else:
    with tempfile.TemporaryDirectory() as td:
        faces = _vision.vision.face_landmarks(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), Path(td))
    if faces is None:
        eye_specs = []
        print("提示: 未检测到人脸，眼下提亮与去痣排除区将不生效")
    else:
        eye_specs = [e for e in faces["eyes"] if e[2] > 0 and e[3] > 0]
        print("Vision 检测眼区: " + ", ".join(
            f"({cx},{cy}) {w}x{h}" for cx, cy, w, h in eye_specs))

# ---------------------------------------------------------------- 1. skin mask
ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]

skin = ((Cr >= 133) & (Cr <= 178) & (Cb >= 77) & (Cb <= 130) & (Y >= 60)).astype(np.uint8)

# keep only the largest connected region (the face+neck); drops stray matches
# on the uniform stripes and background
n, lab, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
if n > 1:
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    skin = (lab == big).astype(np.uint8)

skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
# feather so the edit fades in without a visible border
soft = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 12)
soft = np.clip(soft * 1.6, 0, 1)
print(f"skin covers {100 * (soft > 0.5).mean():.1f}% of frame")

if args.dump_mask:
    cv2.imwrite(args.dump_mask, (soft * 255).astype(np.uint8))

# ------------------------------------------------------------ 2. guided filter
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
L = lab[..., 0]
L_orig = L.copy()   # lab is written in place below; blemish detection needs the original


def guided(I, r, eps):
    """Edge-preserving smoothing of I guided by itself (fast, box-filter based)."""
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mII = cv2.boxFilter(I * I, -1, k)
    var = mII - mI * mI
    a = var / (var + eps)
    b = mI - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


base = guided(L, r=18, eps=1e-3 * 255 * 255)
detail = L - base

# attenuate only the high frequency; edges live in `base` and are untouched
L_new = base + detail * (1.0 - args.smooth) * args.keep_texture

# ------------------------------------------------------------- 3. spot lifting
# compact dark features are lifted toward the local skin tone. The reference is
# a wide blur, so broad shading (under-eye, nose sides) barely registers.
ref = cv2.GaussianBlur(L_new, (0, 0), 45)
deficit = np.clip(ref - L_new, 0, 22)
L_new = L_new + deficit * args.spot

# ------------------------------------------------- 3b. under-eye shadow lift
# The eye bag is broad, soft shading, so the spot lift above deliberately
# ignores it. Here the same deficit idea is applied inside a region derived
# from the eye landmarks -- the lift is proportional to how much darker the
# area is than its surroundings, so it self-limits instead of stamping a
# bright patch under each eye.
if args.undereye > 0 and eye_specs:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ue = np.zeros((H, W), np.float32)
    for cx, cy, ew, eh in eye_specs:
        # sit the ellipse just below the lid: from ~0.35 to ~1.75 eye-heights
        # under the eye centre, so the lid itself is not washed out
        ax, ay = 0.85 * ew, 0.72 * eh
        d = ((xx - cx) / ax) ** 2 + ((yy - (cy + 1.02 * eh)) / ay) ** 2
        ue = np.maximum(ue, np.clip(1.0 - d, 0.0, 1.0))
    ue = cv2.GaussianBlur(np.clip(ue * 2.2, 0, 1) * soft, (0, 0), 12)

    # Reference = local bright envelope of *skin only*: a wide max-filter over
    # the skin gives the surrounding healthy skin tone, so the deficit measures
    # how much darker the bag is than the cheek beside it. Non-skin pixels are
    # zeroed first, and dilate() takes a max, so the white background cannot
    # bleed into the reference.
    L_skin = np.where(soft > 0.02, L_new, 0).astype(np.float32)
    env = cv2.dilate(L_skin, cv2.getStructuringElement(cv2.MORPH_RECT, (151, 151)))
    env = cv2.GaussianBlur(env, (0, 0), 40)
    deficit_ue = np.clip(env - L_new, 0, 60)
    L_new = L_new + deficit_ue * args.undereye * ue
    if args.dump_mask:
        cv2.imwrite(args.dump_mask.replace(".png", "_undereye.png"),
                    (ue * 255).astype(np.uint8))
    print(f"under-eye lift applied to {100 * (ue > 0.5).mean():.2f}% of frame")

# --------------------------------------------------------------- 4. whitening
a_ch, b_ch = lab[..., 1] - 128.0, lab[..., 2] - 128.0
m = soft
L_final = L_new + args.white * m
a_final = a_ch - args.red * m
b_final = b_ch - args.yellow * m

lab[..., 0] = np.clip(L_final, 0, 255)
lab[..., 1] = np.clip(a_final + 128.0, 0, 255)
lab[..., 2] = np.clip(b_final + 128.0, 0, 255)

edited = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

# only skin areas carry the edit; background and clothing stay untouched
m3 = soft[..., None]
out = np.clip(edited * m3 + bgr.astype(np.float32) * (1 - m3), 0, 255).astype(np.uint8)

# ------------------------------------------------------------- 5. mole removal
# Moles are darker *and* more compact than pores, so they survive the smoothing
# above as small round blobs. Find them by a difference-of-gaussians response
# restricted to skin, then inpaint from the surrounding skin.
if args.moles > 0:
    resp = cv2.GaussianBlur(L_orig, (0, 0), 14) - cv2.GaussianBlur(L_orig, (0, 0), 3)
    cand = ((resp > 6) & (soft > 0.5)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, cen = cv2.connectedComponentsWithStats(cand, 8)

    # eyebrows and eyes are excluded geometrically: a false positive inside a
    # brow would inpaint a hole in it. Eyebrows sit up to ~2.5 eye-heights
    # above the eye centre, so the keep-out box is generous.
    keepout = np.zeros((H, W), np.uint8)
    for cx, cy, ew, eh in eye_specs:
        x0 = int(cx - 0.95 * ew); x1 = int(cx + 0.95 * ew)
        y0 = int(cy - 2.6 * eh);  y1 = int(cy + 0.5 * eh)
        keepout[max(y0, 0):y1, max(x0, 0):x1] = 1

    def ray(y0, y1, x0, x1):
        """Mean luminance of a probe patch, or -999 if it is not mostly skin."""
        p, s = L_orig[y0:y1, x0:x1], skin[y0:y1, x0:x1]
        if not p.size or s.sum() < 0.6 * p.size:
            return -999.0
        return float(p.mean())

    spot = np.zeros((H, W), np.uint8)
    kept = 0
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (100 <= area <= 450) or not (8 <= max(w, h) <= 34):
            continue
        if area / (w * h) < 0.55:            # round-ish, not a scratch
            continue
        if max(w, h) / max(min(w, h), 1) > 2.0:
            continue
        if keepout[y:y + h, x:x + w].any():  # in a brow/eye region
            continue

        # A mole is dark on every side with bright skin all around it. Discard
        # anything that sits on a boundary (lip edge, nostril, hairline, ear):
        # those have a dark continuation in at least one direction, which is
        # exactly what the DoG response cannot tell apart on its own.
        cx, cy = int(cen[i][0]), int(cen[i][1])
        r = int(max(w, h) / 2) + 6
        pad = 14
        if cy - r - pad < 0 or cy + r + pad >= H or cx - r - pad < 0 or cx + r + pad >= W:
            continue
        blob = float(L_orig[cc == i].mean())
        neighbours = [
            ray(cy - r - pad, cy - r, cx - 6, cx + 6),        # above
            ray(cy + r, cy + r + pad, cx - 6, cx + 6),        # below
            ray(cy - 6, cy + 6, cx - r - pad, cx - r),        # left
            ray(cy - 6, cy + 6, cx + r, cx + r + pad),        # right
        ]
        if min(neighbours) < blob + 12:
            continue

        spot[cc == i] = 255
        kept += 1
    print(f"moles: {kept} blob(s) detected, inpainting")

    if kept:
        spot = cv2.dilate(spot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        filled = cv2.inpaint(out, spot, 5, cv2.INPAINT_TELEA)
        sf = (spot.astype(np.float32) / 255.0 * args.moles)[..., None]
        out = np.clip(out.astype(np.float32) * (1 - sf) + filled.astype(np.float32) * sf,
                      0, 255).astype(np.uint8)

cv2.imwrite(args.out, out, [cv2.IMWRITE_JPEG_QUALITY, 98])
print("wrote", args.out)
