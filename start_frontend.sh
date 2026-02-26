#!/usr/bin/env bash
# GPUMAP Viz — frontend dev 서버 시작
# Usage: ./start_frontend.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND="$SCRIPT_DIR/frontend"

cd "$FRONTEND"
npm run dev
