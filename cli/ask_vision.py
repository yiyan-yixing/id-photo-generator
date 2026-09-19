#!/usr/bin/env python3
"""Send images plus a question to a local Ollama vision model.

Used as an independent pair of eyes on the ID-photo output: the sub-agents in
this environment cannot render images, so a real local vision model is the only
way to get a second opinion that is actually looking at the picture.

    python3 tools/ask_vision.py "问题" a.png b.jpg ...

Shells out to curl rather than using urllib: in this environment urllib's POSTs
to 127.0.0.1 come back 502 while curl reaches the same endpoint fine, so the
proxy handling is not something to fight.
"""
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL = "qwen3.8-vision:latest"
URL = "http://127.0.0.1:11434/api/chat"


def ask(question: str, images: list[Path], model: str = MODEL,
        timeout: int = 900, think: bool = False) -> str:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": question,
            "images": [base64.b64encode(p.read_bytes()).decode() for p in images],
        }],
        "stream": False,
        "think": think,          # off: the reasoning trace is noise for QC
        "options": {"temperature": 0.2},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        req_path = fh.name
    try:
        proc = subprocess.run(
            ["curl", "-s", "-m", str(timeout), "-X", "POST", URL,
             "-H", "Content-Type: application/json", "--data-binary", f"@{req_path}"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "curl failed")
        body = json.loads(proc.stdout)
        return (body.get("message") or {}).get("content", "").strip()
    finally:
        Path(req_path).unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    q, files = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    t0 = time.time()
    print(ask(q, files))
    print(f"\n[用时 {time.time() - t0:.1f}s · {MODEL}]", file=sys.stderr)
