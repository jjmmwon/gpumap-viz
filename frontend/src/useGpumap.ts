/**
 * useGpumap — WebSocket 연결 및 binary 임베딩 스트리밍 훅
 *
 * Binary frame: header(88B) + flat float32 array
 * Header layout (little-endian):
 * [0..7]   uint64 n_inserted        (insertion ops this iter)
 * [8..11]  uint32 n_points           (total points in embedding)
 * [12..15] uint32 is_done
 * [16..23] uint64 n_update_ops
 * [24..31] uint64 n_embedding_ops
 * [32..35] float32 insertion_time (s)
 * [36..39] float32 update_time    (s)
 * [40..43] float32 embedding_time (s)
 * [44..47] float32 wall_time       (s)
 * [48..55] uint64 insertion_queue_size
 * [56..63] uint64 update_queue_size
 * [64..143] uint64 embedding_queue_levels[10]
 * [144..]   float32 * n_points * 2
 */
import { useCallback, useEffect, useRef, useState } from "react";

const EMBEDDING_QUEUE_LEVELS = 10;
const BINARY_HEADER_BYTES = 64 + EMBEDDING_QUEUE_LEVELS * 8;

function readUint64LEAsNumber(view: DataView, offset: number): number {
  const lo = view.getUint32(offset, true);
  const hi = view.getUint32(offset + 4, true);
  return hi * 2 ** 32 + lo;
}

export interface EmbeddingState {
  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  nInserted: number;
  nPoints: number;
  isDone: boolean;
  key: string;
}

export interface IterationStat {
  wallTime: number;
  insertionTime: number;
  updateTime: number;
  embeddingTime: number;
  insertionOps: number;
  updateOps: number;
  embeddingOps: number;
  insertionQueueSize: number;
  updateQueueSize: number;
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
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const openWebSocket = useCallback(() => {
    if (!mountedRef.current) return;
    // Already open or connecting
    if (wsRef.current && wsRef.current.readyState < WebSocket.CLOSING) return;

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const port = 8000;
    const ws = new WebSocket(`${proto}://${window.location.hostname}:${port}/ws`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    let last = performance.now();

    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        const t0 = performance.now();
        const dtGap = t0 - last;
        last = t0;

        const bytes = ev.data.byteLength;

        const t1 = performance.now();
        console.log(`gap=${dtGap.toFixed(1)}ms handler=${(t1-t0).toFixed(1)}ms bytes=${bytes}`);

        // Binary frame → embedding + iteration stats
        const buf = ev.data as ArrayBuffer;
        if (buf.byteLength < BINARY_HEADER_BYTES) return;
        const view = new DataView(buf);

        const nInserted    = readUint64LEAsNumber(view, 0);
        const nPoints      = view.getUint32(8, true);
        const isDone       = view.getUint32(12, true) !== 0;
        const nUpdateOps   = readUint64LEAsNumber(view, 16);
        const nEmbedOps    = readUint64LEAsNumber(view, 24);
        const insTime      = view.getFloat32(32, true);
        const updTime      = view.getFloat32(36, true);
        const embTime      = view.getFloat32(40, true);
        const wallTime     = view.getFloat32(44, true);
        const insertionQueueSize = readUint64LEAsNumber(view, 48);
        const updateQueueSize = readUint64LEAsNumber(view, 56);
        const embeddingQueueLevels = Array.from(
          { length: EMBEDDING_QUEUE_LEVELS },
          (_, idx) => readUint64LEAsNumber(view, 64 + idx * 8),
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
          insertionQueueSize,
          updateQueueSize,
          embeddingQueueLevels,
        };
        historyRef.current = [...historyRef.current, stat];

        setStatus((prev) => ({
          ...prev,
          phase: isDone ? "done" : "running",
          embedding: { x, y, nInserted, nPoints, isDone, key: `${nInserted}-${wallTime}` },
          history: historyRef.current
        }));

        console.log("Received iteration stat:", stat);
      } else {
        // Text frame → JSON control message
        try {
          const msg = JSON.parse(ev.data as string) as Record<string, unknown>;
          if (msg["type"] === "waiting" || msg["type"] === "ping") {
            // Server is idle or keepalive — ignore
          } else if (msg["type"] === "started") {
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
            // WS stays open — backend returns to waiting state automatically
          } else if (msg["type"] === "error") {
            setStatus((prev) => ({ ...prev, phase: "error", message: msg["message"] as string }));
          }
        } catch {
          /* ignore parse errors */
        }
      }
    };

    ws.onerror = () => {
      setStatus((prev) =>
        prev.phase === "running" ? { ...prev, phase: "error", message: "WebSocket error" } : prev
      );
    };

    ws.onclose = () => {
      // Auto-reconnect after 2s unless unmounted
      if (mountedRef.current) {
        reconnectTimerRef.current = setTimeout(() => openWebSocket(), 2000);
      }
    };
  }, []);

  // Open WebSocket on mount; auto-reconnect on close
  useEffect(() => {
    mountedRef.current = true;
    openWebSocket();
    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on intentional close
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [openWebSocket]);

  const start = useCallback(async (config: GpumapConfig) => {
    historyRef.current = [];
    setStatus({
      phase: "connecting",
      history: [],
      targetLatency: config.target_latency,
      classCategories: undefined,
      classNames: undefined,
    });

    try {
      const resp = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!resp.ok) {
        const text = await resp.text();
        setStatus((prev) => ({ ...prev, phase: "error", message: `Start failed: ${text}` }));
      }
    } catch (err) {
      setStatus((prev) => ({ ...prev, phase: "error", message: `Network error: ${err}` }));
    }
  }, []);

  const stop = useCallback(async () => {
    historyRef.current = [];
    setStatus(INITIAL_STATUS);
    try {
      await fetch("/api/stop", { method: "POST" });
    } catch {
      /* ignore */
    }
  }, []);

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

  // Cleanup on unmount handled in the useEffect above

  return { status, start, stop, uploadNpy };
}
