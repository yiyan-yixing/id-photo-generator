#!/usr/bin/env python3
"""Download the segmentation models used for cutting the subject out.

They are not committed: BiRefNet alone is 928 MB, well past GitHub's 100 MB
per-file limit, and both are reproducible downloads.

HuggingFace proper is not reachable from every network (it is blocked in
mainland China, which is where this was written), so the mirror is tried first
and the official host second. Set HF_ENDPOINT to force one.

    python3 scripts/fetch_models.py            # both
    python3 scripts/fetch_models.py rmbg       # just the fast one
"""
import os
import shutil
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "vendor" / "models"

MODELS = {
    "birefnet": (
        "ZhengPeng7/BiRefNet", "onnx/model.onnx",       # upstream repo
        "onnx-community/BiRefNet-ONNX", "onnx/model.onnx",
        "birefnet.onnx",
    ),
    "rmbg": (
        "briaai/RMBG-1.4", "onnx/model.onnx",
        "briaai/RMBG-1.4", "onnx/model.onnx",
        "rmbg14.onnx",
    ),
}

MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]


def url(endpoint: str, repo: str, path: str) -> str:
    return f"{endpoint.rstrip('/')}/{repo}/resolve/main/{path}"


def download(target: Path, candidates: list[tuple[str, str]]) -> None:
    endpoints = [os.environ["HF_ENDPOINT"]] if os.environ.get("HF_ENDPOINT") else MIRRORS
    errors = []
    for endpoint in endpoints:
        for repo, path in candidates:
            src = url(endpoint, repo, path)
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                print(f"  {src}")
                with urllib.request.urlopen(src, timeout=60) as resp, tmp.open("wb") as out:
                    total = int(resp.headers.get("content-length") or 0)
                    done = 0
                    while chunk := resp.read(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            print(f"\r    {pct:5.1f}%  {done/1e6:7.1f}/{total/1e6:.1f} MB",
                                  end="", flush=True)
                print()
                shutil.move(tmp, target)
                return
            except Exception as exc:                       # noqa: BLE001
                tmp.unlink(missing_ok=True)
                errors.append(f"{src}: {exc}")
    raise SystemExit("  下载失败:\n    " + "\n    ".join(errors))


def main() -> None:
    wanted = sys.argv[1:] or list(MODELS)
    DEST.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        if name not in MODELS:
            raise SystemExit(f"未知模型 {name}，可选：{', '.join(MODELS)}")
        repo_a, path_a, repo_b, path_b, filename = MODELS[name]
        target = DEST / filename
        if target.is_file():
            print(f"[{name}] 已存在，跳过（{target.stat().st_size/1e6:.0f} MB）")
            continue
        print(f"[{name}] 下载中 …")
        download(target, [(repo_a, path_a), (repo_b, path_b)])
        print(f"[{name}] 完成 {target.stat().st_size/1e6:.0f} MB -> {target}")


if __name__ == "__main__":
    main()
