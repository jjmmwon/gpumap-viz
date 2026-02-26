"""
GPUMAPRunner: GPUMAP를 백그라운드 스레드에서 실행하고
asyncio.Queue를 통해 임베딩 결과를 전달합니다.
"""

from __future__ import annotations

import sys
import asyncio
import threading
import traceback
from typing import Optional

import numpy as np

sys.path.insert(0, "/home/jmw/git/proumap/python")

from gpumap import GPUMAP
from gpumap._core import GPUMAPParams
from gpumap.data_loader import DataLoader, PreloadMNISTDataLoader
from gpumap.scheduler import DefaultScheduler
from gpumap.estimator import DefaultLinearEstimator


def _make_data_loader(config: dict) -> DataLoader:
    source = config.get("data_source", "mnist")
    if source == "mnist":
        return PreloadMNISTDataLoader()
    elif source == "npy":
        path = config["data_path"]
        return NpyDataLoader(path)
    else:
        raise ValueError(f"Unknown data_source: {source!r}")


class NpyDataLoader(DataLoader):
    """Load a pre-saved .npy file (shape: [N, D], float32)."""

    def __init__(self, path: str):
        self.X = np.load(path).astype(np.float32)
        if self.X.ndim != 2:
            raise ValueError("npy file must be 2-D array [N, D]")
        self.n_instances = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self._idx = 0

    def get_next_chunk(self, chunk_size: int) -> np.ndarray:
        chunk_size = min(chunk_size, self.n_instances - self._idx)
        chunk = self.X[self._idx : self._idx + chunk_size]
        self._idx += chunk_size
        return chunk


class GPUMAPRunner:
    """
    단일 GPUMAP 실행 세션을 관리합니다.
    WebSocket 핸들러가 asyncio.Queue에서 메시지를 꺼냅니다.

    Queue 메시지 포맷:
        {"type": "started", "n_instances": int, "n_features": int}
        {"type": "embedding", "embedding": np.ndarray[N,2], "n_inserted": int, "is_done": bool,
         "insertion_time": float, "update_time": float, "embedding_time": float}
        {"type": "error", "message": str, "traceback": str}
        {"type": "done"}
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._model: Optional[GPUMAP] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        config: dict,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
    ) -> None:
        with self._lock:
            self._stop_running()
            self._stop_event.clear()
            self._loop = loop
            self._queue = queue
            self._thread = threading.Thread(
                target=self._run,
                args=(config,),
                daemon=True,
                name="gpumap-runner",
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_running()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stop_running(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._close_model()
        self._thread = None

    def _close_model(self) -> None:
        if self._model is not None:
            try:
                self._model.close()
            except Exception:
                pass
            self._model = None

    def _put(self, msg: dict) -> None:
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    def _run(self, config: dict) -> None:
        try:
            data_loader = _make_data_loader(config)

            params = GPUMAPParams()
            params.n_neighbors = int(config.get("n_neighbors", 15))
            params.n_components = 2
            params.min_dist = float(config.get("min_dist", 0.1))
            params.max_epoch = int(config.get("max_epoch", 200))
            params.verbose = bool(config.get("verbose", False))
            params.n_instances = data_loader.n_instances
            params.n_features = data_loader.n_features

            self._model = GPUMAP(
                data_loader=data_loader,
                params=params,
                scheduler=DefaultScheduler(),
                insertion_estimator=DefaultLinearEstimator(),
                update_estimator=DefaultLinearEstimator(),
                embedding_estimator=DefaultLinearEstimator(),
            )

            target_latency = float(config.get("target_latency", 10.0))

            self._put(
                {
                    "type": "started",
                    "n_instances": data_loader.n_instances,
                    "n_features": data_loader.n_features,
                }
            )

            all_inserted = False
            while not self._stop_event.is_set():
                res = self._model.run(target_latency=target_latency)

                # cupy → numpy (GPU→CPU)
                emb: np.ndarray = self._model.get_embedding().get()

                self._put(
                    {
                        "type": "embedding",
                        "embedding": emb,
                        "n_inserted": res.n_insertion_ops,
                        "is_done": res.insertion_completed,
                        "insertion_time": res.insertion_time,
                        "update_time": res.update_time,
                        "embedding_time": res.embedding_time,
                    }
                )

                if res.insertion_completed and not all_inserted:
                    all_inserted = True

        except Exception as exc:
            self._put(
                {
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            self._close_model()
            self._put({"type": "done"})
