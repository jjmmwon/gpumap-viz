"""
FastAPI + WebSocket 서버

엔드포인트:
  POST /api/start   — GPUMAP 실행 시작
  POST /api/stop    — 실행 중지
  GET  /api/status  — 현재 상태 조회
  POST /api/upload  — NPY 파일 업로드
  WS   /ws          — 실시간 임베딩 스트림 (binary)

Binary WebSocket 프레임 포맷 (임베딩):
  Header 76 bytes (little-endian):
    [0..3]   uint32 n_inserted
    [4..7]   uint32 n_points
    [8..11]  uint32 is_done
    [12..15] uint32 n_update_ops
    [16..19] uint32 n_embedding_ops
    [20..23] float32 insertion_time
    [24..27] float32 update_time
    [28..31] float32 embedding_time
    [32..35] float32 wall_time
    [36..75] uint32 embedding_queue_levels[10]
  [76..] float32 * n_points * 2  ← flat [x0,y0,x1,y1,...]
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from runner import GPUMAPRunner

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

runner = GPUMAPRunner()
_current_queue: Optional[asyncio.Queue] = None
_current_config: Optional[dict] = None
_uploaded_npy_path: Optional[str] = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    runner.stop()


app = FastAPI(title="GPUMAP Visualizer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/api/start")
async def api_start(config: dict):
    global _current_queue, _current_config

    # If a run is already in progress, stop it first (run in thread to avoid blocking event loop)
    if runner.is_alive():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, runner.stop)

    _current_config = config
    # maxsize=0 → unbounded: put_nowait never raises QueueFull
    # Runner produces 1 frame per target_latency seconds, WS drains immediately
    _current_queue = asyncio.Queue(maxsize=0)

    # Inject uploaded file path if needed
    if config.get("data_source") == "npy" and _uploaded_npy_path:
        config["data_path"] = _uploaded_npy_path

    loop = asyncio.get_running_loop()
    runner.start(config, loop, _current_queue)

    return {"status": "started", "config": config}


@app.post("/api/stop")
async def api_stop():
    runner.stop()
    return {"status": "stopped"}


@app.get("/api/status")
async def api_status():
    return {
        "running": runner.is_alive(),
        "config": _current_config,
    }


@app.post("/api/upload")
async def api_upload(file: UploadFile):
    global _uploaded_npy_path

    if not file.filename.endswith(".npy"):
        return {"error": "Only .npy files are supported"}

    # Save to a temp file (persists until next upload)
    if _uploaded_npy_path and os.path.exists(_uploaded_npy_path):
        os.remove(_uploaded_npy_path)

    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.close()
    _uploaded_npy_path = tmp.name

    # Probe shape
    arr = np.load(_uploaded_npy_path)
    return {
        "status": "uploaded",
        "filename": file.filename,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if _current_queue is None:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "No run started. Call POST /api/start first."})
        )
        await websocket.close()
        return

    queue = _current_queue  # local ref so we don't miss messages if restarted

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a keepalive ping
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "embedding":
                emb: np.ndarray = msg["embedding"]
                n_inserted: int = msg["n_inserted"]
                is_done: bool = msg["is_done"]
                n_points = emb.shape[0]

                # Binary frame: header(76B) + flat float32 array
                # Header layout (little-endian):
                #   [0..3]   uint32 n_inserted        (insertion ops this iter)
                #   [4..7]   uint32 n_points           (total points in embedding)
                #   [8..11]  uint32 is_done
                #   [12..15] uint32 n_update_ops
                #   [16..19] uint32 n_embedding_ops
                #   [20..23] float32 insertion_time (s)
                #   [24..27] float32 update_time    (s)
                #   [28..31] float32 embedding_time (s)
                #   [32..35] float32 wall_time       (s)
                #   [36..75] uint32 embedding_queue_levels[10]
                #   [76..]   float32 * n_points * 2
                queue_levels = msg.get("embedding_queue_levels", [])
                if not isinstance(queue_levels, list):
                    queue_levels = []
                queue_levels = [int(v) for v in queue_levels[:10]]
                if len(queue_levels) < 10:
                    queue_levels.extend([0] * (10 - len(queue_levels)))
                header = struct.pack(
                    "<IIIIIffff10I",
                    n_inserted,
                    n_points,
                    int(is_done),
                    int(msg.get("n_update_ops", 0)),
                    int(msg.get("n_embedding_ops", 0)),
                    float(msg.get("insertion_time", 0.0)),
                    float(msg.get("update_time", 0.0)),
                    float(msg.get("embedding_time", 0.0)),
                    float(msg.get("wall_time", 0.0)),
                    *queue_levels,
                )
                flat = emb.astype(np.float32, copy=False).flatten()
                await websocket.send_bytes(header + flat.tobytes())

            elif msg_type in ("started", "error", "done", "ping"):
                # Strip non-serialisable values before sending as JSON
                safe = {k: v for k, v in msg.items() if not isinstance(v, np.ndarray)}
                await websocket.send_text(json.dumps(safe))
                if msg_type == "done":
                    break

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Serve frontend in production (optional)
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
