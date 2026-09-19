#!/usr/bin/env python3
"""Local ID-photo service.

Runs on the Mac and is reached from a phone on the same network. Everything
happens on this machine -- the photo is never uploaded anywhere.
"""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from idphoto import pipeline, vision
from idphoto.specs import COLORS, DEFAULT_COLORS, DEFAULT_SIZES, SIZES

# Anything the user can act on comes back as 422 with a readable message;
# unexpected faults stay 500 so they are visibly different.
USER_ERRORS = (pipeline.PipelineError, vision.VisionError)

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
JOBS = BASE / "jobs"
JOB_TTL = 3600               # seconds a generated result stays on disk
MAX_UPLOAD = 40 * 1024 * 1024

# The pipeline is CPU-bound and shells out to Vision; running two at once would
# double peak memory for no real gain on a household tool. Requests queue
# instead, which takes ~2.5s each.
_LOCK = threading.Lock()

app = FastAPI(title="证件照生成器", docs_url=None, redoc_url=None)


def _sweep_jobs() -> None:
    """Drop job directories older than the TTL."""
    if not JOBS.exists():
        return
    cutoff = time.time() - JOB_TTL
    for entry in JOBS.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def _parse_keys(raw: str | None, valid: dict, default: list[str]) -> list[str]:
    if not raw:
        return default
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return [k for k in keys if k in valid] or default


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/specs")
def specs() -> dict:
    """Expose the presets so the page does not duplicate them."""
    return {
        "sizes": [{"key": s.key, "label": s.label, "w": s.w, "h": s.h,
                   "default": s.key in DEFAULT_SIZES}
                  for s in SIZES.values()],
        "colors": [{"key": k, "label": v[0], "rgb": list(v[1]),
                    "default": k in DEFAULT_COLORS}
                   for k, v in COLORS.items()],
        # Most portals cap the upload; the encoder spends the budget on quality.
        "max_kb": [{"key": 0, "label": "不限", "kb": None},
                   {"key": 50, "label": "≤50KB", "kb": 50},
                   {"key": 100, "label": "≤100KB", "kb": 100},
                   {"key": 200, "label": "≤200KB", "kb": 200}],
        "default_max_kb": 0,
        # Two matting models: the stronger one is ~3x slower but keeps far more
        # fine hair. Both are measured; the trade-off is the user's to make.
        "quality": [{"key": "fast", "label": "标准（快）", "secs": "约 9 秒"},
                    {"key": "high", "label": "高精度（慢）", "secs": "约 31 秒"}],
        "default_quality": "high",
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "busy": _LOCK.locked()}


@app.post("/api/process")
def process(
    photo: UploadFile = File(...),
    sizes: str | None = Form(None),
    colors: str | None = Form(None),
    retouch: bool = Form(True),
    max_kb: int | None = Form(None),
    quality: str = Form('high'),
) -> JSONResponse:
    raw = photo.file.read()
    if not raw:
        raise HTTPException(400, "没有收到图片")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "图片太大了，请压缩后再试")

    size_keys = _parse_keys(sizes, SIZES, list(SIZES))
    color_keys = _parse_keys(colors, COLORS, list(COLORS))

    _sweep_jobs()
    job = uuid.uuid4().hex[:12]
    workdir = JOBS / job
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        with _LOCK:
            results, warnings = pipeline.process(
                raw, workdir,
                size_keys=size_keys, color_keys=color_keys,
                do_retouch=retouch, max_kb=max_kb,
                matting_backend=('birefnet' if quality == 'high' else 'rmbg'),
            )
    except USER_ERRORS as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:                       # noqa: BLE001 - report, don't leak a 500 page
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(500, f"处理失败：{exc}") from exc

    for r in results:
        r["url"] = f"/api/result/{job}/{r['filename']}"
    return JSONResponse({
        "job": job,
        "results": results,
        "warnings": warnings,
        # the untouched source, so the page can show what changed
        "original_url": f"/api/result/{job}/original.jpg",
    })


@app.get("/api/result/{job}/{filename}")
def result(job: str, filename: str) -> FileResponse:
    # reject anything that is not a plain name, so `..` cannot escape the job dir
    if Path(filename).name != filename or Path(job).name != job:
        raise HTTPException(400, "非法文件名")
    path = JOBS / job / filename
    if not path.is_file():
        raise HTTPException(404, "结果已过期，请重新生成")
    return FileResponse(path, media_type="image/jpeg", filename=filename)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
