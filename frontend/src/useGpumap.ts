/**
 * useGpumap — WebSocket 연결 및 binary 임베딩 스트리밍 훅
 *
 * Binary 프레임 포맷 (backend/main.py 참고):
 *   Header 76 bytes (little-endian):
 *     [0..3]   uint32 n_inserted
 *     [4..7]   uint32 n_points
 *     [8..11]  uint32 is_done
 *     [12..15] uint32 n_update_ops
 *     [16..19] uint32 n_embedding_ops
 *     [20..23] float32 insertion_time (s)
 *     [24..27] float32 update_time    (s)
 *     [28..31] float32 embedding_time (s)
 *     [32..35] float32 wall_time      (s)
 *     [36..75] uint32 embedding_queue_levels[10]
 *   [76..]   float32 * n_points * 2  (x0,y0, x1,y1, ...)
 */
import { useCallback, useEffect, useRef, useState } from "react";

const EMBEDDING_QUEUE_LEVELS = 10;
const BINARY_HEADER_BYTES = 36 + EMBEDDING_QUEUE_LEVELS * 4;

export interface EmbeddingState {
  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  nInserted: number;
  nPoints: number;
  isDone: boolean;
}

export interface IterationStat {
  wallTime: number;
  insertionTime: number;
  updateTime: number;
  embeddingTime: number;
  insertionOps: number;
  updateOps: number;
  embeddingOps: number;
  embeddingQueueLevels: number[];
}

export interface GpumapStatus {
  phase: "idle" | "connecting" | "running" | "done" | "error";
  message?: string;
  nInstances?: number;
  nFeatures?: number;
  classCategories?: Uint8Array<ArrayBuffer>;
  classNames?: string[];
  embedding?: EmbeddingState;
  history: IterationStat[];
  targetLatency: number;
}

export interface GpumapConfig {
  data_source: "mnist" | "npy";
  n_neighbors: number;
  min_dist: number;
  max_epoch: number;
  target_latency: number;
  verbose?: boolean;
}

/** Split flat [x0,y0,x1,y1,...] Float32Array into separate x/y arrays. */
function splitXY(buf: ArrayBuffer, byteOffset: number, nPoints: number): {
  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
} {
  const x = new Float32Array(nPoints) as Float32Array<ArrayBuffer>;
  const y = new Float32Array(nPoints) as Float32Array<ArrayBuffer>;
  const flat = new Float32Array(buf, byteOffset, nPoints * 2);
  for (let i = 0; i < nPoints; i++) {
    x[i] = flat[i * 2];
    y[i] = flat[i * 2 + 1];
  }
  return { x, y };
}

function encodeClassLabels(raw: unknown[]): { categories: Uint8Array<ArrayBuffer>; names: string[] } | null {
  const values = raw
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));
  if (values.length === 0) return null;

  const unique = Array.from(new Set(values)).sort((a, b) => a - b);
  if (unique.length > 256) return null; // EmbeddingView category type is Uint8Array

  const valueToCategory = new Map<number, number>();
  const names: string[] = [];
  for (let i = 0; i < unique.length; i++) {
    valueToCategory.set(unique[i], i);
    names.push(Number.isInteger(unique[i]) ? `${unique[i]}` : `${unique[i]}`);
  }

  const categories = new Uint8Array(values.length) as Uint8Array<ArrayBuffer>;
  for (let i = 0; i < values.length; i++) {
    const mapped = valueToCategory.get(values[i]);
    categories[i] = mapped ?? 0;
  }

  return { categories, names };
}

const INITIAL_STATUS: GpumapStatus = { phase: "idle", history: [], targetLatency: 10 };

export function useGpumap() {
  const [status, setStatus] = useState<GpumapStatus>(INITIAL_STATUS);
  const wsRef = useRef<WebSocket | null>(null);
  // Mutable accumulator — avoids stale closure issues in onmessage
  const historyRef = useRef<IterationStat[]>([]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const start = useCallback(async (config: GpumapConfig) => {
    disconnect();
    historyRef.current = [];
    setStatus({
      phase: "connecting",
      history: [],
      targetLatency: config.target_latency,
      classCategories: undefined,
      classNames: undefined,
    });

    // 1. POST /api/start
    try {
      const resp = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!resp.ok) {
        const text = await resp.text();
        setStatus((prev) => ({ ...prev, phase: "error", message: `Start failed: ${text}` }));
        return;
      }
    } catch (err) {
      setStatus((prev) => ({ ...prev, phase: "error", message: `Network error: ${err}` }));
      return;
    }

    // 2. Open WebSocket
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus((prev) => ({ ...prev, phase: "running" }));
    };

    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        // Binary frame → embedding + iteration stats
        const buf = ev.data as ArrayBuffer;
        if (buf.byteLength < BINARY_HEADER_BYTES) return;
        const view = new DataView(buf);

        const nInserted    = view.getUint32(0,  true);
        const nPoints      = view.getUint32(4,  true);
        const isDone       = view.getUint32(8,  true) !== 0;
        const nUpdateOps   = view.getUint32(12, true);
        const nEmbedOps    = view.getUint32(16, true);
        const insTime      = view.getFloat32(20, true);
        const updTime      = view.getFloat32(24, true);
        const embTime      = view.getFloat32(28, true);
        const wallTime     = view.getFloat32(32, true);
        const embeddingQueueLevels = Array.from(
          { length: EMBEDDING_QUEUE_LEVELS },
          (_, idx) => view.getUint32(36 + idx * 4, true),
        );
        const expectedBytes = BINARY_HEADER_BYTES + nPoints * 2 * 4;
        if (buf.byteLength < expectedBytes) return;
        const { x, y } = splitXY(buf, BINARY_HEADER_BYTES, nPoints);

        const stat: IterationStat = {
          wallTime,
          insertionTime: insTime,
          updateTime: updTime,
          embeddingTime: embTime,
          insertionOps: nInserted,
          updateOps: nUpdateOps,
          embeddingOps: nEmbedOps,
          embeddingQueueLevels,
        };
        historyRef.current = [...historyRef.current, stat];

        setStatus((prev) => ({
          ...prev,
          phase: isDone ? "done" : "running",
          embedding: { x, y, nInserted, nPoints, isDone },
          history: historyRef.current,
        }));
      } else {
        // Text frame → JSON control message
        try {
          const msg = JSON.parse(ev.data as string) as Record<string, unknown>;
          if (msg["type"] === "started") {
            const encodedClass = Array.isArray(msg["class_labels"])
              ? encodeClassLabels(msg["class_labels"] as unknown[])
              : null;
            setStatus((prev) => ({
              ...prev,
              phase: "running",
              nInstances: msg["n_instances"] as number,
              nFeatures: msg["n_features"] as number,
              classCategories: encodedClass?.categories,
              classNames: encodedClass?.names,
            }));
          } else if (msg["type"] === "done") {
            setStatus((prev) => ({ ...prev, phase: "done" }));
            disconnect();
          } else if (msg["type"] === "error") {
            setStatus((prev) => ({ ...prev, phase: "error", message: msg["message"] as string }));
            disconnect();
          }
          // ping → ignore
        } catch {
          /* ignore parse errors */
        }
      }
    };

    ws.onerror = () => {
      setStatus((prev) => ({ ...prev, phase: "error", message: "WebSocket error" }));
    };

    ws.onclose = () => {
      setStatus((prev) =>
        prev.phase === "running" ? { ...prev, phase: "done" } : prev
      );
    };
  }, [disconnect]);

  const stop = useCallback(async () => {
    disconnect();
    setStatus(INITIAL_STATUS);
    historyRef.current = [];
    try {
      await fetch("/api/stop", { method: "POST" });
    } catch {
      /* ignore */
    }
  }, [disconnect]);

  const uploadNpy = useCallback(async (file: File): Promise<{ shape: number[]; dtype: string } | null> => {
    const form = new FormData();
    form.append("file", file);
    try {
      const resp = await fetch("/api/upload", { method: "POST", body: form });
      if (!resp.ok) return null;
      return await resp.json() as { shape: number[]; dtype: string };
    } catch {
      return null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => () => { disconnect(); }, [disconnect]);

  return { status, start, stop, uploadNpy };
}
