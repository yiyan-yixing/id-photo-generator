"""Image loading plus wrappers around the two macOS Vision helper binaries.

`segment` produces a soft-alpha matte via VNGenerateForegroundInstanceMaskRequest
and `facebox` prints face landmarks via VNDetectFaceLandmarksRequest. Both work on
raw pixels and ignore EXIF, so images must be orientation-normalised before either
is called.
"""
from __future__ import annotations

import io
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TOOLS = Path(__file__).resolve().parent.parent / "vendor" / "tools"
SEGMENT = TOOLS / "segment"
FACEBOX = TOOLS / "facebox"

# Only genuinely huge images get downscaled. The cap sits above 4032 so that a
# stock 12MP iPhone photo passes through untouched rather than taking a needless
# resample.
MAX_DIM = 5000


class VisionError(RuntimeError):
    """Raised when a Vision helper fails in a way the caller should report."""


def ensure_built() -> None:
    """Compile the Swift helpers if the binaries are missing or stale."""
    for name in ("segment", "facebox"):
        binary, source = TOOLS / name, TOOLS / f"{name}.swift"
        if not source.exists():
            raise VisionError(f"缺少源码 {source}")
        if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
            continue
        proc = subprocess.run(
            ["swiftc", "-O", str(source), "-o", str(binary)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise VisionError(f"编译 {name} 失败: {proc.stderr.strip()[:400]}")


def _sips_to_jpeg(raw: bytes) -> Image.Image:
    """Fallback decoder for formats Pillow cannot read (HEIC from iPhone)."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.bin"
        dst = Path(td) / "out.jpg"
        src.write_bytes(raw)
        proc = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(src), "--out", str(dst)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not dst.exists():
            raise VisionError("无法识别这张图片的格式，请换一张 JPG 或 PNG")
        return Image.open(dst).convert("RGB")


def load_image(raw: bytes) -> np.ndarray:
    """Decode to an EXIF-corrected RGB array, capped at MAX_DIM on the long side."""
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = ImageOps.exif_transpose(im) or im
        im = im.convert("RGB")
    except Exception:
        im = _sips_to_jpeg(raw)

    w, h = im.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.array(im)


def _run(binary: Path, args: list[str], no_result_msg: str | None = None) -> str:
    if not binary.exists():
        ensure_built()
    proc = subprocess.run([str(binary), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if no_result_msg and "no foreground" in detail:
            raise VisionError(no_result_msg)
        raise VisionError(detail[:400] or "Vision 调用失败")
    return proc.stdout


def _write_temp(rgb: np.ndarray, directory: Path, name: str = "src.jpg") -> Path:
    path = directory / name
    # callers hand over float arrays straight from the pipeline; Pillow wants bytes
    arr = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, "JPEG", quality=98, subsampling=0)
    return path


def segment(rgb: np.ndarray, workdir: Path) -> np.ndarray:
    """Foreground matte as float32 in [0, 1], same HxW as `rgb`."""
    src = _write_temp(rgb, workdir)
    mask = workdir / "mask.png"
    _run(SEGMENT, [str(src), str(mask)],
         no_result_msg="没能在照片里找到人物，请换一张单人正面照")
    if not mask.exists():
        raise VisionError("前景分割没有输出结果，请换一张照片")
    alpha = np.array(Image.open(mask).convert("L"), dtype=np.float32) / 255.0

    if alpha.shape != rgb.shape[:2]:     # Vision should echo the input size
        alpha = np.array(
            Image.fromarray((alpha * 255).astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    if alpha.max() < 0.5:
        raise VisionError("没能在照片里找到人物，请换一张单人正面照")
    return alpha


_FACE_BBOX = re.compile(r"face bbox px: x=(-?\d+) y=(-?\d+) w=(-?\d+) h=(-?\d+)")
_CHIN = re.compile(r"chin apex: x=(-?\d+) y=(-?\d+)")
_EYE = re.compile(
    r"(\w+Eye) centre: x=(-?\d+) y=(-?\d+)\s+xrange (-?\d+)\.\.(-?\d+)\s+yrange (-?\d+)\.\.(-?\d+)"
)


def face_landmarks(rgb: np.ndarray, workdir: Path) -> dict | None:
    """Vision face box, chin apex and per-eye boxes. None when no face is found."""
    src = _write_temp(rgb, workdir)
    out = _run(FACEBOX, [str(src)])

    box = _FACE_BBOX.search(out)
    chin = _CHIN.search(out)
    eyes = _EYE.findall(out)
    if not box or not chin or not eyes:
        return None

    def box_of(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        return ((x0 + x1) // 2, (y0 + y1) // 2, abs(x1 - x0), abs(y1 - y0))

    # the regex captures (name, cx, cy, xmin, xmax, ymin, ymax) -- note the
    # min/max pairs are per-axis, so they must be regrouped before use
    return {
        "face_box": tuple(int(v) for v in box.groups()),
        "chin": (int(chin.group(1)), int(chin.group(2))),
        "eyes": [box_of(int(e[3]), int(e[5]), int(e[4]), int(e[6])) for e in eyes],
    }
