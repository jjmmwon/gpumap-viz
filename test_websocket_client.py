#!/usr/bin/env python3
"""
Test client for GPUMAP Viz backend.

Flow:
  1) POST /api/start with JSON config (like frontend)
  2) Connect WS /ws and receive:
      - text frames: started/error/done/ping
      - binary frames: header(76B) + float32 embedding (n_points*2)

Usage:
  python ws_start_and_recv.py --base http://localhost:8000
"""

import argparse
import asyncio
import json
import struct
import time
from typing import Any, Dict, Optional

import aiohttp

HEADER_FMT = "<IIIIIffff10I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 76 bytes


DEFAULT_CONFIG: Dict[str, Any] = {
    "data_source": "mnist",
    "n_neighbors": 15,
    "min_dist": 0.1,
    "max_epoch": 200,
    "target_latency": 3.0,
    "verbose": False,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="http://localhost:8000", help="e.g. http://localhost:8000")
    p.add_argument("--ws", type=str, default=None, help="Override WS URL, e.g. ws://localhost:8000/ws")
    p.add_argument("--config", type=str, default=None, help="JSON string to override config")
    p.add_argument("--frames", type=int, default=0, help="Stop after N binary frames (0=never)")
    return p.parse_args()


def make_ws_url(base_http: str) -> str:
    # http://host:8000 -> ws://host:8000/ws
    if base_http.startswith("https://"):
        return "wss://" + base_http[len("https://"):] + "/ws"
    if base_http.startswith("http://"):
        return "ws://" + base_http[len("http://"):] + "/ws"
    # fallback
    return "ws://" + base_http + "/ws"


async def post_start(session: aiohttp.ClientSession, base: str, config: Dict[str, Any]) -> None:
    url = base.rstrip("/") + "/api/start"
    print(f"[client] POST {url}")
    async with session.post(url, json=config) as resp:
        text = await resp.text()
        if resp.status // 100 != 2:
            raise RuntimeError(f"Start failed: HTTP {resp.status} body={text[:300]}")
        print(f"[client] start ok: HTTP {resp.status} body={text[:200] if text else '(empty)'}")


def parse_binary_frame(data: bytes) -> Dict[str, Any]:
    if len(data) < HEADER_SIZE:
        return {"ok": False, "reason": "too_small", "size": len(data)}

    header = data[:HEADER_SIZE]
    (
        n_inserted,
        n_points,
        is_done,
        n_update_ops,
        n_embedding_ops,
        insertion_time,
        update_time,
        embedding_time,
        wall_time,
        *queue_levels,
    ) = struct.unpack(HEADER_FMT, header)

    payload_bytes = len(data) - HEADER_SIZE
    expected = int(n_points) * 2 * 4
    ok = payload_bytes == expected

    return {
        "ok": ok,
        "size": len(data),
        "n_points": int(n_points),
        "n_inserted": int(n_inserted),
        "is_done": int(is_done),
        "n_update_ops": int(n_update_ops),
        "n_embedding_ops": int(n_embedding_ops),
        "insertion_time": float(insertion_time),
        "update_time": float(update_time),
        "embedding_time": float(embedding_time),
        "wall_time": float(wall_time),
        "queue_levels": list(map(int, queue_levels)),
        "payload_bytes": int(payload_bytes),
        "expected_payload_bytes": int(expected),
    }


async def recv_ws(ws: aiohttp.ClientWebSocketResponse, max_frames: int = 0) -> None:
    frames = 0
    bytes_total = 0
    t0 = time.perf_counter()

    async for msg in ws:
        now = time.perf_counter()

        if msg.type == aiohttp.WSMsgType.TEXT:
            text = msg.data
            # 서버가 JSON string으로 보냄
            try:
                obj = json.loads(text)
            except Exception:
                obj = None

            if isinstance(obj, dict):
                t = obj.get("type")
                if t == "ping":
                    # 서버 ping은 그냥 무시
                    continue
                # print(f"[ws/text] {obj[:200]}")
                if t == "done":
                    print("[client] got done, closing.")
                    return
                if t == "error":
                    print("[client] got error, closing.")
                    return
            else:
                print(f"[ws/text] {text[:200]}")

        elif msg.type == aiohttp.WSMsgType.BINARY:
            data = msg.data
            frames += 1
            bytes_total += len(data)

            info = parse_binary_frame(data)
            dt = now - t0
            mb = bytes_total / (1024 * 1024)
            rate = (mb / dt) if dt > 0 else 0.0

            if info.get("ok"):
                print(
                    f"[ws/bin] frame#{frames} size={info['size']/1024:.1f}KB "
                    f"n_points={info['n_points']} is_done={info['is_done']} "
                    f"rate={rate:.2f}MB/s"
                )
            else:
                print(f"[ws/bin] frame#{frames} BAD frame: {info}")

            if max_frames > 0 and frames >= max_frames:
                print(f"[client] reached max_frames={max_frames}, closing.")
                return

        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            print("[client] ws closed/error")
            return


async def main() -> None:
    args = parse_args()

    base = args.base.rstrip("/")
    ws_url = args.ws or make_ws_url(base)

    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(json.loads(args.config))

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1) start
        await post_start(session, base, config)

        # 2) ws connect
        print(f"[client] WS connect {ws_url}")
        async with session.ws_connect(
            ws_url,
            autoclose=True,
            autoping=True,
            heartbeat=30.0,
            max_msg_size=0,  # unlimited
        ) as ws:
            await recv_ws(ws, max_frames=args.frames)


if __name__ == "__main__":
    asyncio.run(main())