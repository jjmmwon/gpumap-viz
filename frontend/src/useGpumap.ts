/**
 * useGpumap — WebSocket 연결 및 binary 임베딩 스트리밍 훅
 *
 * Binary 프레임 포맷 (backend/main.py 참고):
 *   [uint32 n_inserted][uint32 n_points][uint8 is_done]
 *   [float32 * n_points * 2]  ← 평탄화된 [x0,y0, x1,y1, ...]
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface EmbeddingState {
  x: Float32Array<ArrayBuffer>;
  y: Float32Array<ArrayBuffer>;
  nInserted: number;
  nPoints: number;
  isDone: boolean;
}

export interface GpumapStatus {
  phase: "idle" | "connecting" | "running" | "done" | "error";
  message?: string;
  nInstances?: number;
  nFeatures?: number;
  embedding?: EmbeddingState;
}

export interface GpumapConfig {
  data_source: "mnist" | "npy";
  n_neighbors: number;
  min_dist: number;
  n_epochs: number;
  target_latency: number;
  verbose?: boolean;
}

/** Split flat [x0,y0,x1,y1,...] Float32Array into separate x/y arrays.
 *  The input float32 view shares the original ArrayBuffer, so we copy into
 *  fresh ArrayBuffers to satisfy Float32Array<ArrayBuffer> typing. */
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

export function useGpumap() {
  const [status, setStatus] = useState<GpumapStatus>({ phase: "idle" });
  const wsRef = useRef<WebSocket | null>(null);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const start = useCallback(async (config: GpumapConfig) => {
    disconnect();
    setStatus({ phase: "connecting" });

    // 1. POST /api/start
    try {
      const resp = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!resp.ok) {
        const text = await resp.text();
        setStatus({ phase: "error", message: `Start failed: ${text}` });
        return;
      }
    } catch (err) {
      setStatus({ phase: "error", message: `Network error: ${err}` });
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
        // Binary frame → embedding
        const buf = ev.data as ArrayBuffer;
        const view = new DataView(buf);
        const nInserted = view.getUint32(0, true);
        const nPoints = view.getUint32(4, true);
        const isDone = view.getUint8(8) !== 0;
        // byteOffset=9: after the 9-byte header
        const { x, y } = splitXY(buf, 9, nPoints);

        setStatus((prev) => ({
          ...prev,
          phase: isDone ? "done" : "running",
          embedding: { x, y, nInserted, nPoints, isDone },
        }));
      } else {
        // Text frame → JSON control message
        try {
          const msg = JSON.parse(ev.data as string) as Record<string, unknown>;
          if (msg["type"] === "started") {
            setStatus((prev) => ({
              ...prev,
              phase: "running",
              nInstances: msg["n_instances"] as number,
              nFeatures: msg["n_features"] as number,
            }));
          } else if (msg["type"] === "done") {
            setStatus((prev) => ({ ...prev, phase: "done" }));
            disconnect();
          } else if (msg["type"] === "error") {
            setStatus({ phase: "error", message: msg["message"] as string });
            disconnect();
          }
          // ping → ignore
        } catch {
          /* ignore parse errors */
        }
      }
    };

    ws.onerror = () => {
      setStatus({ phase: "error", message: "WebSocket error" });
    };

    ws.onclose = () => {
      setStatus((prev) =>
        prev.phase === "running" ? { ...prev, phase: "done" } : prev
      );
    };
  }, [disconnect]);

  const stop = useCallback(async () => {
    disconnect();
    setStatus({ phase: "idle" });
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
