#!/bin/bash
# Start the ID-photo service and print the address to open on your phone.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY="${PY:-python3}"

# Two Swift helpers wrap the macOS Vision framework (foreground matting fallback
# and face landmarks). Compile them from source on first run.
for tool in segment facebox; do
  src="vendor/tools/$tool.swift"
  bin="vendor/tools/$tool"
  if [ ! -x "$bin" ] || [ "$src" -nt "$bin" ]; then
    echo "编译 $tool …"
    swiftc -O "$src" -o "$bin"
  fi
done

# The matting models are ~1.1 GB and are not committed; fetch on first run.
if [ ! -f vendor/models/birefnet.onnx ] && [ ! -f vendor/models/rmbg14.onnx ]; then
  echo "首次运行：下载抠图模型（约 1.1 GB，可用 HF_ENDPOINT 指定镜像）"
  "$PY" scripts/fetch_models.py
fi

IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '')"
echo
echo "  证件照生成器已启动"
echo "  手机浏览器打开:  http://${IP:-<本机IP>}:${PORT}"
echo "  本机访问:        http://127.0.0.1:${PORT}"
echo
echo "  手机需与本机连同一个 WiFi。按 Ctrl-C 停止。"
echo

exec "$PY" -m uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
