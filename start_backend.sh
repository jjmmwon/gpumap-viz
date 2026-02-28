#!/usr/bin/env bash
# GPUMAP Viz — backend 서버 시작
# Usage: ./start_backend.sh [port]
set -e

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"

cd "$BACKEND"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[error] conda environment is not activated. Run: conda activate gpumap"
  exit 1
fi

echo "[env] CONDA_PREFIX=$CONDA_PREFIX"
echo "[env] python=$(which python)"
python -V

echo "[start] Backend at http://localhost:$PORT"
uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
