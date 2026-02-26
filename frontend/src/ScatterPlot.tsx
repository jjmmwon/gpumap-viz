/**
 * ScatterPlot - embedding-atlas EmbeddingView wrapper
 * https://apple.github.io/embedding-atlas/embedding-view.html
 */
import { useState } from "react";
import { EmbeddingView } from "embedding-atlas/react";
import type { DataPoint } from "embedding-atlas/react";
import type { EmbeddingState } from "./useGpumap";

interface Props {
  embedding?: EmbeddingState;
}

export function ScatterPlot({ embedding }: Props) {
  const [tooltip, setTooltip] = useState<DataPoint | null>(null);

  if (!embedding || embedding.nPoints === 0) {
    return (
      <div style={{
        width: "100%",
        height: "100%",
        border: "1px dashed #cfdae8",
        borderRadius: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--muted)",
        fontSize: 15,
        flexDirection: "column",
        gap: 10,
        background: "var(--surface-alt)",
      }}>
        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4">
          <circle cx="12" cy="12" r="10" />
          <circle cx="8" cy="10" r="1.5" fill="currentColor" />
          <circle cx="14" cy="8" r="1.5" fill="currentColor" />
          <circle cx="10" cy="15" r="1.5" fill="currentColor" />
          <circle cx="16" cy="14" r="1.5" fill="currentColor" />
        </svg>
        <span>Press START to stream points</span>
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100%", position: "relative", overflow: "hidden", background: "var(--surface)" }}>
      <EmbeddingView
        data={{ x: embedding.x, y: embedding.y }}
        tooltip={tooltip}
        onTooltip={setTooltip}
      />
    </div>
  );
}
