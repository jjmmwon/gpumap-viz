"""
GPUMAPRunner: GPUMAP를 백그라운드 스레드에서 실행하고
asyncio.Queue를 통해 임베딩 결과를 전달합니다.
"""

from __future__ import annotations

import sys
import asyncio
import resource
import threading
import traceback
import time
from typing import Optional

import numpy as np

from gpumap import GPUMAP
from gpumap._core import GPUMAPParams
from gpumap.data_loader import DataLoader, PreloadMNISTDataLoader
from gpumap.policy import DefaultWeightedPolicy


_MAX_NEIGHBORS = 32
_NUM_SAMPLES = 32


def _is_bad_alloc_error(exc: Exception) -> bool:
    return isinstance(exc, MemoryError) or "std::bad_alloc" in str(exc).lower()


def _format_bad_alloc_message(exc: Exception) -> str:
    detail = str(exc).strip() or "std::bad_alloc"
    lower = detail.lower()
    if (
        "cudaerroroperatingsystem" in lower
        or "operation not supported on this os" in lower
    ):
        return (
            "CUDA runtime initialization failed (cudaErrorOperatingSystem). "
            "This process cannot allocate GPU memory. "
            f"Original error: {detail}. "
            "Run this in the same environment to verify: "
            '`python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"`'
        )
    return f"GPUMAP failed with std::bad_alloc. Original error: {detail}"


def _try_raise_memlock_limit() -> None:
    """RLIMIT_MEMLOCK을 unlimited로 올린다.

    GPUMAP 재시작 시 이전 인스턴스의 pinned memory(~44MB)가 OS에 아직
    반납되지 않은 상태에서 새 인스턴스가 동일 크기를 요청하면 합산
    ~88MB가 필요해져 기본 한도 64MiB를 초과 → std::bad_alloc이 발생한다.
    """
    rlim_inf = getattr(resource, "RLIM_INFINITY", -1)

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft in (-1, rlim_inf):
        return  # 이미 unlimited — 출력 불필요

    # soft/hard 모두 unlimited로 올리기 시도 (root 또는 CAP_SYS_RESOURCE 필요)
    try:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (rlim_inf, rlim_inf))
        print("[gpumap] RLIMIT_MEMLOCK → unlimited", flush=True)
        return
    except (ValueError, OSError):
        pass

    # hard limit 범위 내에서 soft만 올리기 시도
    if hard not in (-1, rlim_inf):
        try:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        except (ValueError, OSError):
            pass

    # 현재 실제 soft 값 다시 읽기
    soft_now, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft_now in (-1, rlim_inf):
        print("[gpumap] RLIMIT_MEMLOCK → unlimited", flush=True)
        return

    limit_mb = soft_now / (1024 * 1024)
    print(
        f"[gpumap] 경고: RLIMIT_MEMLOCK={limit_mb:.0f} MiB (hard={hard / (1024*1024):.0f} MiB) — "
        "재시작 시 bad_alloc 발생 가능.\n"
        "         영구 해결: /etc/security/limits.conf 에 아래 두 줄 추가 후 재로그인:\n"
        "           *  soft  memlock  unlimited\n"
        "           *  hard  memlock  unlimited",
        flush=True,
    )


def _log_gpu_state(attempt: int) -> None:
    """GPUMAP 생성 직전 GPU/시스템 상태를 stderr에 출력한다."""
    import cupy as cp

    try:
        free, total = cp.cuda.runtime.memGetInfo()
        free_mb = free / (1024 * 1024)
        total_mb = total / (1024 * 1024)
    except Exception as e:
        free_mb = total_mb = -1
        print(f"[gpumap-debug] memGetInfo failed: {e}", flush=True)

    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    unlimited = soft in (-1, getattr(resource, "RLIM_INFINITY", -1))
    memlock_str = "unlimited" if unlimited else f"{soft / (1024*1024):.1f} MiB"

    print(
        f"[gpumap-debug] attempt={attempt}"
        f" | GPU free={free_mb:.0f} MB / total={total_mb:.0f} MB"
        f" | RLIMIT_MEMLOCK soft={memlock_str}",
        flush=True,
    )


def _check_cuda_runtime() -> None:
    # Fail fast with an actionable message instead of a generic bad_alloc later.
    import cupy as cp

    for attempt in range(2):
        try:
            n_devices = int(cp.cuda.runtime.getDeviceCount())
            if n_devices <= 0:
                raise RuntimeError("No CUDA devices detected.")
            return
        except Exception as exc:
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise RuntimeError(
                "CUDA runtime check failed before GPUMAP init: " f"{exc}"
            ) from exc


def _check_limits(config: dict, n_instances: int) -> None:
    n_neighbors = int(config.get("n_neighbors", 15))
    if n_neighbors < 1 or n_neighbors > _MAX_NEIGHBORS:
        raise ValueError(
            f"n_neighbors must be in [1, {_MAX_NEIGHBORS}] (got {n_neighbors})."
        )

    # Approximate pinned-memory use from NND buffers in C++.
    # 5 pinned matrices [N, NUM_SAMPLES] of int32 + pinned int2/int vectors.
    estimated_pinned_bytes = (
        n_instances * _NUM_SAMPLES * 4 * 5 + n_instances * 8 * 2 + n_instances * 4
    )

    soft_limit, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    unlimited = soft_limit in (-1, getattr(resource, "RLIM_INFINITY", -1))
    if not unlimited and estimated_pinned_bytes > int(soft_limit):
        est_mb = estimated_pinned_bytes / (1024 * 1024)
        lim_mb = int(soft_limit) / (1024 * 1024)
        raise RuntimeError(
            "Pinned host-memory requirement is likely above RLIMIT_MEMLOCK. "
            f"Estimated pinned bytes: {est_mb:.1f} MiB, limit: {lim_mb:.1f} MiB. "
            "Reduce dataset size, or increase memlock (e.g. `ulimit -l unlimited`) "
            "before starting backend."
        )


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
        {"type": "started", "n_instances": int, "n_features": int, "class_labels": list[int] (optional)}
        {"type": "embedding", "embedding": np.ndarray[N,2], "n_inserted": int, "is_done": bool,
         "insertion_time": float, "update_time": float, "embedding_time": float,
         "embedding_queue_levels": list[int]}
        {"type": "error", "message": str, "traceback": str}
        {"type": "done"}
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        # Per-thread stop event: each _run() receives its own Event so that
        # creating a new event for a new thread never accidentally un-signals
        # the still-running old thread.
        self._thread_stop: Optional[threading.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._model: Optional[GPUMAP] = None
        self._lock = threading.Lock()  # guards start/stop sequencing
        self._model_lock = threading.Lock()  # guards self._model access

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
            had_previous_run = self._thread is not None
            self._stop_running()
            if had_previous_run:
                # WSL2에서 cudaFreeHost로 반환된 pinned memory(~62MB)가
                # WDDM 드라이버 레이어에 실제로 반납되기까지 latency가 있음.
                # 즉시 cudaMallocHost를 시도하면 std::bad_alloc이 발생하므로
                # 짧게 대기해서 OS가 페이지를 회수할 시간을 준다.
                time.sleep(1.0)
            self._loop = loop
            self._queue = queue
            stop_event = threading.Event()
            self._thread_stop = stop_event
            self._thread = threading.Thread(
                target=self._run,
                args=(config, stop_event),
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
        if self._thread_stop is not None:
            self._thread_stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        # Force-close only when the thread is still alive (join timed out).
        # If it exited cleanly, its finally block already closed the model.
        if self._thread is not None and self._thread.is_alive():
            with self._model_lock:
                if self._model is not None:
                    try:
                        self._model.close()
                    except Exception:
                        pass
                    self._model = None
        self._thread = None
        self._thread_stop = None

    def _put(self, msg: dict) -> None:
        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    def _run(self, config: dict, stop_event: threading.Event) -> None:
        model = None
        try:
            _try_raise_memlock_limit()
            _check_cuda_runtime()
            data_loader = _make_data_loader(config)
            _check_limits(config, data_loader.n_instances)

            params = GPUMAPParams()
            params.n_neighbors = int(config.get("n_neighbors", 15))
            params.n_components = 2
            params.min_dist = float(config.get("min_dist", 0.1))
            params.max_epoch = int(config.get("max_epoch", 200))
            params.verbose = bool(config.get("verbose", False))
            params.n_instances = data_loader.n_instances
            params.n_features = data_loader.n_features

            # WSL2에서 CUDA 런타임 첫 초기화 시 _core.GPUMAP() 또는
            # _impl.initialize() 에서 std::bad_alloc이 발생할 수 있음.
            # gpumap.py는 _core.GPUMAP() 만 retry하므로 여기서 전체를 retry.
            for attempt in range(2):
                try:
                    _log_gpu_state(attempt)
                    model = GPUMAP(
                        data_loader=data_loader,
                        params=params,
                        policy=DefaultWeightedPolicy(),
                    )
                    with self._model_lock:
                        self._model = model
                    break
                except Exception as exc:
                    if not _is_bad_alloc_error(exc):
                        raise
                    if attempt == 0:
                        time.sleep(4)
                        continue
                    raise RuntimeError(_format_bad_alloc_message(exc)) from exc

            target_latency = float(config.get("target_latency", 10.0))

            class_labels: Optional[list[int]] = None
            # PreloadMNISTDataLoader exposes `y` labels aligned with insertion order.
            raw_labels = getattr(data_loader, "y", None)
            if raw_labels is not None:
                try:
                    labels_arr = np.asarray(raw_labels).reshape(-1)
                    if labels_arr.shape[0] == data_loader.n_instances:
                        class_labels = labels_arr.astype(np.int32, copy=False).tolist()
                except Exception:
                    class_labels = None

            self._put(
                {
                    "type": "started",
                    "n_instances": data_loader.n_instances,
                    "n_features": data_loader.n_features,
                    "class_labels": class_labels,
                }
            )

            assert model is not None
            all_inserted = False
            iter = 0
            while not stop_event.is_set():
                iter += 1

                t0 = time.perf_counter()
                res = model.run(target_latency=target_latency)

                # cupy → numpy (GPU→CPU)
                emb: np.ndarray = model.get_embedding().get()

                wall_time = time.perf_counter() - t0

                print(
                    f"[gpumap-debug] run loop iteration {iter} : wall_time={wall_time:.2f}s"
                )

                if res.insertion_completed and not all_inserted:
                    all_inserted = True

                embedding_queue_levels = list(
                    getattr(res, "embedding_queue_level_sizes", [])
                )
                embedding_queue_empty = sum(embedding_queue_levels) == 0

                is_done = all_inserted and embedding_queue_empty
                self._put(
                    {
                        "type": "embedding",
                        "embedding": emb,
                        "n_inserted": res.n_insertion_ops,
                        "n_update_ops": res.n_update_ops,
                        "n_embedding_ops": res.n_embedding_ops,
                        "is_done": is_done,
                        "insertion_time": res.insertion_time,
                        "update_time": res.update_time,
                        "embedding_time": res.embedding_time,
                        "wall_time": wall_time,
                        "insertion_queue_size": getattr(res, "insertion_queue_size", 0),
                        "update_queue_size": getattr(res, "update_queue_size", 0),
                        "embedding_queue_levels": embedding_queue_levels,
                    }
                )

                if is_done:
                    break

        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self._put(
                {
                    "type": "error",
                    "message": message,
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            # Close only the model this thread created; never touch self._model
            # if it has already been replaced by a new session.
            if model is not None:
                try:
                    model.close()
                except Exception:
                    pass
                with self._model_lock:
                    if self._model is model:
                        self._model = None
            self._put({"type": "done"})
