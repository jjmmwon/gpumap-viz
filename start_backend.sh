#!/usr/bin/env bash
# GPUMAP Viz — backend 서버 시작
# Usage: ./start_backend.sh [port]
set -e

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"

cd "$BACKEND"

if [ ! -f ".venv/bin/activate" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

echo "[start] Backend at http://localhost:$PORT"
uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
