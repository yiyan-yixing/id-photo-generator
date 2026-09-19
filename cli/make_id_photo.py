#!/usr/bin/env python3
"""Build a student ID photo from a normalized portrait + Vision alpha mask.

Framing follows the common Chinese school e-photo convention: 480x640 (3:4),
head (crown->chin) about 64% of frame height, ~9% clear space above the crown,
head horizontally centred on the face midline.

Landmarks (crown / chin / midline) are detected by Vision when not passed on the
command line. Pass them only to override that; the pixel coordinates are tied to
the image's grid, so hand-typed values go stale as soon as the input changes.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps

import _vision

OUT_W, OUT_H = 480, 640          # required 3:4 target
BG = (255, 255, 255)             # white background

p = argparse.ArgumentParser()
p.add_argument("src")
# optional: the mask must be in the SAME orientation as `src`, and Vision's
# segmentation ignores EXIF, so a mask made from an un-rotated phone photo would
# be sideways here. Omitting it and letting this script segment the already
# orientation-corrected image is the safe default; pass a path to reuse one.
p.add_argument("mask", nargs="?", default=None,
               help="alpha mask PNG; generated with Vision when omitted")
p.add_argument("out")
p.add_argument("--crown", type=int, default=None,
               help="crown row; detected from the matte if omitted")
p.add_argument("--chin", type=int, default=None,
               help="chin row; detected by Vision if omitted")
p.add_argument("--midline", type=int, default=None,
               help="face vertical axis; detected by Vision if omitted")
p.add_argument("--head-frac", type=float, default=0.64)
p.add_argument("--crown-margin", type=float, default=0.09)
p.add_argument("--enhance", type=float, default=0.0,
               help="0 = untouched, 1 = full gentle lift")
p.add_argument("--dump-raw", default=None,
               help="also write the full-resolution composite here (for retouching)")
p.add_argument("--edge-guide", type=float, default=0.0,
               help="guided-filter radius to snap the matte onto real image edges")
p.add_argument("--edge-blur", type=float, default=0.0,
               help="sigma used to de-ragged the matte before remapping")
p.add_argument("--edge-width", type=float, default=0.30,
               help="half-width of the alpha transition after remapping")
args = p.parse_args()

# exif_transpose matters: Vision reads raw pixels, so an un-rotated iPhone photo
# would put the landmarks in a different coordinate space than the mask.
src_img = Image.open(args.src)
img = (ImageOps.exif_transpose(src_img) or src_img).convert("RGB")
a = np.array(img).astype(np.float32)
H, W = a.shape[:2]

if args.mask:
    mask_img = Image.open(args.mask)
    mask_img = ImageOps.exif_transpose(mask_img) or mask_img
    m = np.array(mask_img.convert("L").resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
else:
    with tempfile.TemporaryDirectory() as td:
        print("未提供蒙版，用 Vision 分割…")
        m = _vision.vision.segment(a, Path(td))

# --- landmarks: ask Vision unless the caller pinned them -------------------
if args.crown is None or args.chin is None or args.midline is None:
    with tempfile.TemporaryDirectory() as td:
        faces = _vision.vision.face_landmarks(a, Path(td))
    if faces is None:
        raise SystemExit("没检测到人脸；如果是手动指定，请传入 --chin/--midline")
    eyes = [e for e in faces["eyes"] if e[2] > 0 and e[3] > 0]
    if not eyes:
        raise SystemExit("没能定位到眼睛，请换一张清晰的正面照")

    if args.chin is None:
        args.chin = int(faces["chin"][1])
    if args.midline is None:
        eye_mid = sum(e[0] for e in eyes) / len(eyes)
        args.midline = int(round((eye_mid + faces["chin"][0]) / 2))
    if args.crown is None:
        # top of the hair, taken where the silhouette becomes solid -- the
        # topmost mask pixels are stray flyaway strands a few px wide
        args.crown = _vision.matte.crown_row(m, faces["face_box"][2])
    print(f"Vision 检测: crown={args.crown} chin={args.chin} midline={args.midline}")

if args.chin <= args.crown:
    raise SystemExit(f"头部范围异常 (crown={args.crown}, chin={args.chin})")

# --- matte refinement ------------------------------------------------------
# Same algorithm the service uses, so manual output matches the app. Pass
# --edge-guide to fall back to the older guided-filter smoothing for comparison.
if args.edge_guide > 0:
    # legacy path: guided filter with the image as guide, then a blur+remap that
    # narrows the transition band
    guide = cv2.GaussianBlur(
        cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5) / 255.0
    r = int(args.edge_guide)
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(guide, -1, k)
    mP = cv2.boxFilter(m, -1, k)
    cov = cv2.boxFilter(guide * m, -1, k) - mI * mP
    var = cv2.boxFilter(guide * guide, -1, k) - mI * mI
    aa = cov / (var + 1e-4)
    bb = mP - aa * mI
    m = cv2.boxFilter(aa, -1, k) * guide + cv2.boxFilter(bb, -1, k)
    if args.edge_blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), args.edge_blur)
        t = np.clip((m - 0.5) / (2 * args.edge_width) + 0.5, 0, 1)
        m = t * t * (3 - 2 * t)
else:
    m = _vision.matte.refine_matte(a, m)
m = np.clip(m, 0, 1)

# --- crop box --------------------------------------------------------------
crop_h = int(round((args.chin - args.crown) / args.head_frac / 4)) * 4
crop_w = crop_h * OUT_W // OUT_H
top = args.crown - int(round(crop_h * args.crown_margin))
left = args.midline - crop_w // 2

if crop_h > H or crop_w > W:
    raise SystemExit(
        f"裁切框 {crop_w}x{crop_h} 超出图像 {W}x{H}：照片留白不足，"
        f"请换一张头部占比更小的照片，或调低 --head-frac")
top = min(max(top, 0), H - crop_h)
left = min(max(left, 0), W - crop_w)
box = (left, top, left + crop_w, top + crop_h)

print(f"crop {crop_w}x{crop_h} at {box[:2]}  ratio {crop_h/crop_w:.4f}")
print(f"head {args.chin-args.crown}px = {100*(args.chin-args.crown)/crop_h:.1f}% of frame height")

# --- composite subject onto background ------------------------------------
subj = a[top:top + crop_h, left:left + crop_w].copy()
alpha = m[top:top + crop_h, left:left + crop_w][..., None]

if args.enhance > 0:
    k = args.enhance
    x = np.clip(subj / 255.0, 0, 1)
    x = np.power(x, 1.0 - 0.06 * k)            # lift midtones
    x = x + 0.06 * k * np.sin(np.pi * x)       # gentle midtone contrast
    subj = np.clip(x * 255.0, 0, 255)

bg = np.array(BG, dtype=np.float32).reshape(1, 1, 3)
out = subj * alpha + bg * (1.0 - alpha)

im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

if args.dump_raw:
    im.save(args.dump_raw, "JPEG", quality=98, subsampling=0)
    print("wrote raw", args.dump_raw, im.size)

im = im.resize((OUT_W, OUT_H), Image.LANCZOS)
im.save(args.out, "JPEG", quality=92, subsampling=0, optimize=True)
print("wrote", args.out, im.size)
