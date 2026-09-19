"""End-to-end pipeline: uploaded photo bytes -> ID photos for every size/colour."""
from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import matte as matte_mod
from . import matting
from . import vision
from .retouch import RetouchParams, retouch, skin_mask
from .specs import COLORS, SIZES, Size

# Framing: head height as a fraction of the frame, and the clear space left
# above the crown. Both are ratios, so the vertical crop is identical across
# every output size -- only the width changes with the aspect ratio.
HEAD_FRAC = 0.64
CROWN_MARGIN = 0.09

JPEG_QUALITY = 98      # used when no size budget is set


class PipelineError(RuntimeError):
    """A failure we can explain to the user."""


@dataclass
class Landmarks:
    crown: int
    chin: int
    midline: int
    eyes: list[tuple[int, int, int, int]]   # cx, cy, w, h per eye


def _resolve_landmarks(alpha: np.ndarray, faces: dict) -> Landmarks:
    face_x, _, face_w, _ = faces["face_box"]
    chin_x, chin_y = faces["chin"]
    eyes = [e for e in faces["eyes"] if e[2] > 0 and e[3] > 0]
    if not eyes:
        raise PipelineError("没能定位到眼睛，请换一张清晰的正面照")

    # Midline from the eye centres is more stable than the chin apex, which
    # shifts with head tilt. Average the two for a slight robustness gain.
    eye_mid = sum(e[0] for e in eyes) / len(eyes)
    midline = int(round((eye_mid + chin_x) / 2))
    crown = matte_mod.crown_row(alpha, face_w)
    if chin_y <= crown:
        raise PipelineError("检测到的头部范围异常，请换一张照片")
    return Landmarks(crown=crown, chin=int(chin_y), midline=midline, eyes=eyes)


def _global_lift(rgb: np.ndarray, alpha: np.ndarray, target_median: float = 150.0,
                 max_lift: float = 0.08) -> np.ndarray:
    """Lift a flat, dull capture without touching a photo that is already bright.

    A fixed gamma would blow out highlights on a well-exposed photo, so the
    strength scales with how far the median luminance sits below the target and
    is capped. Applied in L only, to avoid shifting hue.

    The median is taken over the SUBJECT, not the whole frame: a head-and-
    shoulders crop is mostly background, so a whole-frame median says more about
    where the subject sits against the wall than about how the subject is lit.
    """
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    subject = alpha > 0.5
    if subject.sum() < 100:
        return rgb
    med = float(np.median(lab[..., 0][subject]))
    k = float(np.clip((target_median - med) / target_median, 0.0, 1.0))
    if k <= 0:
        return rgb
    amt = max_lift * k
    x = np.clip(lab[..., 0] / 255.0, 0, 1)
    lab[..., 0] = np.clip((np.power(x, 1.0 - amt) + amt * np.sin(np.pi * x)) * 255.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _estimate_background(rgb: np.ndarray, alpha: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """Smooth per-pixel estimate of the ORIGINAL background.

    Normalised convolution: blur the image weighted by how background-y each
    pixel is, then divide by the blurred weights. The subject contributes no
    weight, so this extrapolates the wall across the hair.
    """
    w = (1.0 - alpha)[..., None].astype(np.float32)
    num = cv2.GaussianBlur(rgb.astype(np.float32) * w, (0, 0), sigma)
    # keep the weights 3-D: GaussianBlur would collapse an (H, W, 1) array to 2-D
    den = cv2.GaussianBlur(np.repeat(w, 3, axis=2), (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def _unmix(subject: np.ndarray, background: np.ndarray, alpha: np.ndarray,
           cap: float = 1.25) -> np.ndarray:
    """Recover the true foreground colour from a matted pixel.

    The observed pixel is `F*a + B*(1-a)`, so a soft matte alone is not enough:
    compositing the observed pixel onto a new colour leaks the ORIGINAL
    background through the semi-transparent fringe (an error of
    `a(1-a)(B_old - F)`, tens of levels on dark hair). Dividing it back out
    removes that term.

    The division is unstable where alpha is small, so the result is capped
    against a local envelope of the recovered foreground. Without it, the last
    few percent of a bright wall pixel left in the matte inflates the division
    into near-white, and the result is a thin pale fringe tracing the whole
    silhouette -- exactly the cut-out look we are trying to avoid. The cap is
    relative, not absolute, so genuinely lighter hair is still allowed through.

    Tuned against the crown rim on a red background (mean G in the silhouette
    band, where a pure red background reads 0): 1.25 gives 42.5, 1.0 gives 39.6.
    The extra 7% is not worth taking 1.0: the envelope is a local mean, so near
    the jaw it is dragged down by neighbouring dark hair, and a tighter cap then
    clips the face itself. Verified -- at 1.0 the measured skin tone falls to
    b*=10.1 (grey) and apparent texture to 38%.
    """
    a = np.maximum(alpha, 0.2)[..., None]          # clamp: low-alpha pixels carry no weight
    f = np.clip((subject.astype(np.float32) - background * (1.0 - alpha)[..., None]) / a,
                0, 255)

    if cap >= 1.0:
        envelope = matte_mod._normconv(f.mean(axis=2), (alpha > 0.8).astype(np.float32), 30.0)
        f = np.minimum(f, np.maximum(envelope * cap, 40.0)[..., None])

    # A solid pixel is already its own foreground: there is nothing to divide
    # out, and doing it anyway hurts. The division is exact only when alpha is
    # trustworthy, and the matte is never certain everywhere -- around the eyes,
    # for instance, it sits below 1 over the lid. Dividing there drags the lid's
    # chroma toward the background's neutral and it starts reading as a grey-green
    # eyeshadow. Measured on the reference photo: the lid's G-R moved from -21.4
    # to -15.9 in this step alone, while every other stage left it within 2.
    solid = alpha > 0.95
    f = np.where(solid[..., None], subject.astype(np.float32), f)
    return f


def _fill_edge_with_hair(subject: np.ndarray, alpha: np.ndarray,
                         threshold: float = 0.95) -> np.ndarray:
    """Repaint the semi-transparent fringe with the hair's own colour.

    Un-mixing is only as good as its alpha, and on the fringe alpha is never
    exact. Writing the estimate as a ratio `k = a_true/a_est`, the composite
    error works out to

        error = a_true * (1 - k) / k * (original background - new background)

    which is positive whenever alpha is over-estimated -- i.e. the fringe comes
    out LIGHTER than the hair it is supposed to be, and a pale rim traces the
    whole silhouette. That rim is what reads as an unnatural edge, and it does
    not go away by tuning the matte: any estimate error re-creates it.

    So stop re-deriving the colour there. Hair is close to a single tone, and the
    interior gives it directly, so sample that tone and paint the fringe with it.
    This is the automated form of what a retoucher does by cloning from inside
    the hair outward along the edge, and it removes the rim outright instead of
    shrinking it.
    """
    conf = (alpha > 0.97).astype(np.float32)
    lab = cv2.cvtColor(subject.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    tone = np.stack([matte_mod._normconv(lab[..., i], conf, 25.0) for i in range(3)], axis=2)
    tone_rgb = cv2.cvtColor(np.clip(tone, 0, 255).astype(np.uint8),
                            cv2.COLOR_LAB2RGB).astype(np.float32)
    return np.where((alpha < threshold)[..., None], tone_rgb, subject.astype(np.float32))


def _lift_shadows(rgb: np.ndarray, alpha: np.ndarray, below: float,
                  amount: float = 15.0, feather: float = 0.05) -> np.ndarray:
    """Raise the dark end of the subject so clothing is not a black mass.

    Whitening brightens the skin, which leaves the uniform and the shadowed
    neck looking disproportionately dark beside the face. The gain tapers off
    towards the highlights, so nothing blows out -- only the shadows move.

    Restricted to below the chin. Hair is dark too, and lifting it just greys it
    out and costs the strand contrast that makes it read as hair; the uniform is
    what actually needs the lift. `below` is the chin row as a fraction of the
    height, and the cutoff is feathered so there is no visible seam.
    """
    if amount <= 0:
        return rgb
    h = rgb.shape[0]
    rows = np.arange(h, dtype=np.float32).reshape(-1, 1)
    cut = below * h
    m = np.clip((rows - cut) / max(feather * h, 1.0), 0.0, 1.0)
    m = m * np.clip(alpha, 0, 1)

    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = np.clip(lab[..., 0] / 255.0, 0, 1)
    lab[..., 0] = np.clip(lab[..., 0] + amount * np.power(1.0 - lum, 1.5) * m, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _crop_padded(arr: np.ndarray, box: tuple[int, int, int, int], fill) -> np.ndarray:
    """Crop `box`, padding with `fill` where the box leaves the image.

    Padding rather than clamping keeps the requested aspect ratio exact:
    an off-centre or very tight portrait would otherwise force a crop that no
    longer matches the spec. Padded regions are background, so they come out as
    flat colour.
    """
    x0, y0, x1, y1 = box
    H, W = arr.shape[:2]
    shape = (y1 - y0, x1 - x0) + arr.shape[2:]
    out = np.empty(shape, dtype=arr.dtype)
    out[...] = fill

    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, W), min(y1, H)
    if sx0 < sx1 and sy0 < sy1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def _save_preview(rgb: np.ndarray, path: Path, max_side: int = 900) -> None:
    """Write a downscaled copy of the source for side-by-side comparison."""
    im = Image.fromarray(rgb)
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                       Image.LANCZOS)
    im.save(path, "JPEG", quality=86, subsampling=0)


def _check_skin_tone(subject: np.ndarray, alpha: np.ndarray) -> list[str]:
    """Guard the whitening against going past what real skin looks like.

    In CIELAB, skin sits around b* +12..+25. Below that the face reads grey
    rather than fair, which is the usual failure mode of over-whitening, and
    exactly the kind of thing an automated check is likely to reject.

    Measured over the skin only -- the subject also contains hair and clothing,
    which are legitimately low-chroma and would drag any whole-figure average
    far below the skin range.
    """
    m = (skin_mask(subject) > 0.8) & (alpha > 0.99)
    if m.sum() < 500:
        return []
    lab = cv2.cvtColor(subject.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = float(np.median(lab[..., 0][m]))
    b_star = float(np.mean(lab[..., 2][m]) - 128.0)

    out = []
    if b_star < 12.0:
        out.append(f"肤色偏灰（b*={b_star:.1f}，正常应≥12），建议调低美白")
    if lum > 200.0:
        out.append(f"肤色过亮（L={lum:.0f}），可能影响人脸核验")
    return out


def _check_texture(before: np.ndarray, after: np.ndarray, alpha: np.ndarray,
                   soft: np.ndarray) -> list[str]:
    """Flag smoothing that has gone past what skin still looks like.

    Skin reads as plastic once the fine detail is gone, and that is a different
    failure from a wrong tone -- the geometry is untouched, so the landmark check
    cannot see it. The measurable proxy is how much of the high-frequency energy
    survives: a light retouch keeps most of it, a heavy one leaves a waxy
    surface. Below roughly half is where retouchers say it stops looking like
    skin, so the threshold is 0.55.

    Calibrated on the reference photo (fraction of high-frequency energy kept):
    smoothing alone 80%, the shipped defaults 66%, a heavy smooth + full spot
    lift 46%. 0.55 sits between the default and the heavy end, so the shipped
    settings have margin and only a deliberate over-smooth trips it.
    """
    m = (soft > 0.8) & (alpha > 0.99)
    if m.sum() < 2000:
        return []

    def detail(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.5))[m].mean())

    d0, d1 = detail(before), detail(after)
    if d0 < 0.5:
        return []
    kept = d1 / d0
    if kept < 0.55:
        return [f"皮肤质感损失较多（仅保留 {kept * 100:.0f}%），可能显得不自然，建议调低磨皮"]
    return []


def _check_face_consistency(subject: np.ndarray, lm: Landmarks, size: Size,
                            offset: tuple[int, int],
                            supersample: float = 1.0) -> list[str]:
    """Re-run the Vision face detector on the finished crop.

    Framing is derived from landmarks, so if the retouch had distorted the face
    the detector would either lose it outright or place it somewhere else. Both
    mean the photo is likely to fail automated verification downstream.

    `offset` is where the crop starts in the ORIGINAL image's x axis -- the
    landmarks are in original coordinates, so both the working crop's origin and
    the sub-crop's own shift have to come out before scaling.
    """
    oh, ow = subject.shape[:2]
    # `subject` is on the (possibly enlarged) working canvas while the landmarks
    # and the offset are in original-image pixels, so bring the canvas back to
    # original scale before comparing -- otherwise the check reports a large
    # false drift as soon as supersampling is on.
    scale_x = size.w / (ow / supersample)
    scale_y = size.h / (oh / supersample)
    offset_x, offset_y = offset
    expected_x = sorted((cx - offset_x) * scale_x for cx, _, _, _ in lm.eyes)
    mean_cy = sum(cy for _, cy, _, _ in lm.eyes) / len(lm.eyes)
    expected_y = (mean_cy - offset_y) * scale_y

    with tempfile.TemporaryDirectory() as td:
        try:
            found = vision.face_landmarks(subject, Path(td))
        except vision.VisionError:
            found = None
    if not found or len(found["eyes"]) != len(expected_x):
        return ["完成图里已检测不到人脸，可能修图过头，建议关闭精修后重试"]

    # the detector ran on the working canvas, so bring its coordinates back to
    # original-image pixels first -- scale_x is expressed against the original
    # crop width, and skipping the division inflates the measured positions by
    # the supersample factor and reports a large false drift
    got_x = sorted(e[0] / supersample * scale_x for e in found["eyes"])
    got_y = (sum(e[1] for e in found["eyes"]) / len(found["eyes"])
             / supersample * scale_y)
    drift = max(max(abs(g - e) for g, e in zip(got_x, expected_x)),
                abs(got_y - expected_y)) / size.h
    if drift > 0.02:
        return [f"人脸位置较原图偏移约 {drift * 100:.1f}%，可能无法通过核验"]
    return []


def _crop_box(lm: Landmarks, size: Size) -> tuple[int, int, int, int]:
    crop_h = int(round((lm.chin - lm.crown) / HEAD_FRAC))
    crop_w = int(round(crop_h / size.aspect))
    top = lm.crown - int(round(crop_h * CROWN_MARGIN))
    left = lm.midline - crop_w // 2
    return (left, top, left + crop_w, top + crop_h)


def render(subject: np.ndarray, alpha: np.ndarray, size: Size,
           color: tuple[int, int, int], max_kb: int | None = None) -> bytes:
    """Scale the already-cropped subject to `size` and composite onto `color`."""
    # Resize premultiplied colour and alpha with the same filter. Doing it this
    # way (rather than resizing the colour alone) is what keeps the fringe
    # correct: `F*a + bg*(1-a)` needs both terms scaled together.
    pre = cv2.resize(subject.astype(np.float32) * alpha[..., None], (size.w, size.h),
                     interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(alpha.astype(np.float32), (size.w, size.h),
                     interpolation=cv2.INTER_AREA)[..., None]

    bg = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = np.clip(pre + bg * (1.0 - a_s), 0, 255).astype(np.uint8)
    return _encode_jpeg(Image.fromarray(out), size.dpi, max_kb)


def _encode_jpeg(im: Image.Image, dpi: int, max_kb: int | None) -> bytes:
    """Encode the JPEG, spending as much quality as the size budget allows.

    Submission portals routinely cap the upload (100KB is common), and the
    right response is to use the best quality that still fits rather than to
    pick one fixed quality and hope. Chroma subsampling is the last resort --
    it is what makes coloured edges muddy, which is exactly where these photos
    are sensitive, so quality is reduced first.
    """
    def encode(quality: int, subsampling: int) -> bytes:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, subsampling=subsampling,
                optimize=True, dpi=(dpi, dpi))
        return buf.getvalue()

    if max_kb is None:
        return encode(JPEG_QUALITY, 0)

    budget = max_kb * 1024
    best = encode(40, 0)
    if len(best) > budget:
        # Even at the floor it does not fit: only subsampling can help now.
        return encode(40, 2) if len(encode(40, 2)) <= budget else best

    # Binary search the highest quality that stays inside the budget. The cap is
    # a ceiling to spend up to, not a target to aim at -- a portal that accepts
    # 100KB should get the best 100KB photo, so the results stay monotonic in the
    # budget (an uncapped output is not smaller than a capped one).
    lo, hi = 40, JPEG_QUALITY
    while lo < hi:
        mid = (lo + hi + 1) // 2
        data = encode(mid, 0)
        if len(data) <= budget:
            best, lo = data, mid
        else:
            hi = mid - 1
    return best


def process(raw: bytes, workdir: Path, *, size_keys=None, color_keys=None,
            do_retouch: bool = True, max_kb: int | None = None,
            matting_tiles: int = 2, matting_backend: str = matting.DEFAULT_BACKEND,
            params: RetouchParams | None = None,
            supersample: float = 1.5,
            max_work_pixels: int = 18_000_000) -> tuple[list[dict], list[str]]:
    """Run the whole pipeline. Returns (results, warnings)."""
    size_keys = [k for k in (size_keys or SIZES) if k in SIZES] or list(SIZES)
    color_keys = [k for k in (color_keys or COLORS) if k in COLORS] or list(COLORS)

    vision.ensure_built()
    rgb = vision.load_image(raw)
    warnings_early: list[str] = []

    # Keep a copy of the source so the page can put it next to the results --
    # without it there is no way to judge whether the background swap looks
    # natural. Scaled down: it is only ever a reference thumbnail.
    _save_preview(rgb, workdir / "original.jpg")

    faces = vision.face_landmarks(rgb, workdir)
    if faces is None:
        raise PipelineError("没检测到人脸，请换一张正面免冠照")

    # The matte comes from a matting model where one is available. The Vision
    # segmentation is kept as the fallback, but it mislabels interior background
    # as subject and no amount of edge work downstream can repair that -- see
    # matting.py for the measurement.
    if matting.available(matting_backend):
        alpha = matting.predict(rgb, tiles=matting_tiles, backend=matting_backend)
        backend = "rmbg"
    else:
        warnings_early.append("未找到抠图模型，已退回系统分割（边缘质量可能下降）")
        alpha = matte_mod.refine_matte(rgb, vision.segment(rgb, workdir))
        backend = "vision"
    lm = _resolve_landmarks(alpha, faces)

    # Every output shares the same vertical framing and is centred on the face,
    # so the widest requested size's box contains all the others. Work inside
    # that box and the retouch runs exactly once, at a consistent scale -- the
    # filter radii in retouch.py are tuned in pixels, so retouching at the full
    # sensor resolution would change how large a "pore" or an eye bag is.
    widest = max((SIZES[k] for k in size_keys), key=lambda s: s.margin)
    box = _crop_box(lm, widest)
    H, W = alpha.shape

    warnings: list[str] = []
    if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
        warnings.append("照片留白不足，已用底色补齐边缘")
    warnings = warnings_early + warnings

    rgb_c = _crop_padded(rgb, box, fill=(0, 0, 0))
    alpha_c = _crop_padded(alpha[..., None], box, fill=(0.0,))[..., 0]
    ox, oy = box[0], box[1]
    eyes_c = [(cx - ox, cy - oy, w, h) for cx, cy, w, h in lm.eyes]
    below = (lm.chin - oy) / alpha_c.shape[0]

    # Work on a larger canvas, then come back down at the end. The matte steps
    # (1px contract, 1px blur, the narrow remap) are in PIXELS, so at the working
    # resolution they are coarser than one output pixel once the final downscale
    # is applied -- the hair fringe comes out as visible stair-steps. Enlarging
    # first makes every one of those operations sub-pixel at the delivered size,
    # and the closing downscale averages away the single-pixel colour noise the
    # un-mixing leaves behind. Measured on the hair fringe: 1x reads blocky,
    # 2x resolves the strands, 3x is marginally finer for 2x the runtime.
    if supersample > 1.0:
        h, w = rgb_c.shape[:2]
        # Memory grows with the square of the factor -- 2x on this crop peaked at
        # 4.5 GB against 2.7 GB at 1.5x -- so the enlargement is capped by a
        # canvas budget rather than trusted blindly on a large source.
        room = (max_work_pixels / float(h * w)) ** 0.5
        eff = max(1.0, min(supersample, room))
        nw, nh = int(round(w * eff)), int(round(h * eff))
        rgb_c = cv2.resize(rgb_c, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        alpha_c = cv2.resize(alpha_c, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        eyes_c = [(cx * eff, cy * eff, w * eff, h * eff) for cx, cy, w, h in eyes_c]
        if eff < supersample - 0.05:
            warnings.append(f"原图较大，已按内存上限把处理精度调整为 {eff:.1f}x")
        supersample = eff

    # Lift before the retouch, so the skin mask and the whitening below both
    # operate on the same tones the final image will have.
    rgb_c = _global_lift(rgb_c, alpha_c)

    if do_retouch:
        white = (rgb_c.astype(np.float32) * alpha_c[..., None]
                 + 255.0 * (1 - alpha_c[..., None]))
        subject, stats = retouch(white.astype(np.uint8), rgb_c, alpha_c, eyes_c,
                                 params or RetouchParams())
    else:
        subject = rgb_c

    # chin as a fraction of the working crop, so the lift starts at the neck
    subject = _lift_shadows(subject, alpha_c, below=below)

    # Peel the original background out of the semi-transparent fringe before
    # compositing onto the requested colours.
    subject = _unmix(subject, _estimate_background(rgb_c, alpha_c), alpha_c)
    # ...then stop trusting that estimate out on the fringe, where the alpha it
    # divides by is least reliable, and paint the hair's own tone instead.
    subject = _fill_edge_with_hair(subject, alpha_c)

    # Guard the result before it is offered to the user: the whitening has to
    # stay inside what real skin looks like, and the face has to survive intact.
    if do_retouch:
        warnings.extend(_check_skin_tone(subject, alpha_c))
        warnings.extend(_check_texture(rgb_c, subject, alpha_c, skin_mask(rgb_c)))

    # A narrower aspect is a centred sub-rectangle of the working crop. The
    # midline comes from the landmarks in original-image pixels, but everything
    # below indexes the working canvas, so convert it -- mixing the two units
    # silently offsets every narrower size.
    midline_c = int(round((lm.midline - ox) * supersample))
    results = []
    for sk in size_keys:
        size = SIZES[sk]
        sub_w = int(round(alpha_c.shape[0] / size.aspect))
        sub_w = min(sub_w, alpha_c.shape[1])
        sub_left = midline_c - sub_w // 2
        sub_left = max(0, min(sub_left, alpha_c.shape[1] - sub_w))
        if do_retouch and sk == size_keys[0]:
            warnings.extend(_check_face_consistency(
                subject[:, sub_left:sub_left + sub_w], lm, size,
                (ox + sub_left / supersample, oy), supersample))
        for ck in color_keys:
            label, color = COLORS[ck]
            data = render(subject[:, sub_left:sub_left + sub_w],
                          alpha_c[:, sub_left:sub_left + sub_w], size, color, max_kb)
            results.append({
                "size": sk, "color": ck,
                "label": f"{size.label} · {label}",
                "size_label": size.label, "color_label": label,
                "w": size.w, "h": size.h,
                "filename": f"{sk}_{ck}.jpg",
                "bytes": len(data),
            })
            try:
                (workdir / f"{sk}_{ck}.jpg").write_bytes(data)
            except OSError as exc:
                raise PipelineError(f"写入结果失败: {exc}") from exc

    return results, warnings
