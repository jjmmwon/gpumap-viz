/**
 * ScatterPlot - embedding-atlas EmbeddingView wrapper
 * https://apple.github.io/embedding-atlas/embedding-view.html
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { EmbeddingView } from "embedding-atlas/react";
import type { DataPoint } from "embedding-atlas/react";
import type { EmbeddingState } from "./useGpumap";

interface Props {
  embedding?: EmbeddingState;
  classCategories?: Uint8Array<ArrayBuffer>;
  classNames?: string[];
}

const TABLEAU10 = [
  "#4E79A7",
  "#F28E2B",
  "#E15759",
  "#76B7B2",
  "#59A14F",
  "#EDC948",
  "#B07AA1",
  "#FF9DA7",
  "#9C755F",
  "#BAB0AC",
];

export function ScatterPlot({ embedding, classCategories, classNames }: Props) {
  const [tooltip, setTooltip] = useState<DataPoint | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [viewSize, setViewSize] = useState({ width: 0, height: 0 });
  const nPoints = embedding?.nPoints ?? 0;

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    const updateSize = (width: number, height: number) => {
      const nextWidth = Math.max(1, Math.floor(width));
      const nextHeight = Math.max(1, Math.floor(height));
      setViewSize((prev) =>
        prev.width === nextWidth && prev.height === nextHeight
          ? prev
          : { width: nextWidth, height: nextHeight },
      );
    };

    updateSize(element.clientWidth, element.clientHeight);

    if (typeof ResizeObserver === "undefined") {
      const onResize = () => updateSize(element.clientWidth, element.clientHeight);
      window.addEventListener("resize", onResize);
      return () => window.removeEventListener("resize", onResize);
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      updateSize(entry.contentRect.width, entry.contentRect.height);
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const visibleCategories = useMemo(() => {
    if (!classCategories || nPoints <= 0) return undefined;
    const nVisible = Math.min(nPoints, classCategories.length);
    if (nVisible <= 0) return undefined;
    return classCategories.subarray(0, nVisible) as Uint8Array<ArrayBuffer>;
  }, [classCategories, nPoints]);

  const categoryCount = useMemo(() => {
    let maxCategory = -1;
    if (visibleCategories) {
      for (let i = 0; i < visibleCategories.length; i++) {
        if (visibleCategories[i] > maxCategory) maxCategory = visibleCategories[i];
      }
    }
    return Math.max(classNames?.length ?? 0, maxCategory + 1);
  }, [classNames, visibleCategories]);

  const categoryColors = useMemo(() => {
    if (categoryCount <= 0) return null;
    return Array.from({ length: categoryCount }, (_, idx) => TABLEAU10[idx % TABLEAU10.length]);
  }, [categoryCount]);

  const legendItems = useMemo(() => {
    if (!visibleCategories || !categoryColors) return [];
    const present = new Set<number>();
    for (let i = 0; i < visibleCategories.length; i++) {
      present.add(visibleCategories[i]);
    }
    return Array.from(present)
      .sort((a, b) => a - b)
      .map((category) => ({
        category,
        color: categoryColors[category],
        label: classNames?.[category] ?? `${category}`,
      }));
  }, [classNames, categoryColors, visibleCategories]);

  const embeddingData =
    embedding && embedding.nPoints > 0
      ? (visibleCategories
        ? { x: embedding.x, y: embedding.y, category: visibleCategories }
        : { x: embedding.x, y: embedding.y })
      : null;

  const showPlaceholder = !embedding || embedding.nPoints === 0;

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        overflow: "hidden",
        background: "var(--surface)",
      }}
    >
      {showPlaceholder ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            border: "1px dashed #cfdae8",
            borderRadius: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--muted)",
            fontSize: 16,
            flexDirection: "column",
            gap: 10,
            background: "var(--surface-alt)",
          }}
        >
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4">
            <circle cx="12" cy="12" r="10" />
            <circle cx="8" cy="10" r="1.5" fill="currentColor" />
            <circle cx="14" cy="8" r="1.5" fill="currentColor" />
            <circle cx="10" cy="15" r="1.5" fill="currentColor" />
            <circle cx="16" cy="14" r="1.5" fill="currentColor" />
          </svg>
          <span>Press START to stream points</span>
        </div>
      ) : (
        viewSize.width > 0 &&
        viewSize.height > 0 &&
        embeddingData && (
          <EmbeddingView
            data={embeddingData}
            categoryColors={categoryColors}
            width={viewSize.width}
            height={viewSize.height}
            tooltip={tooltip}
            key={embedding.key}

            onTooltip={setTooltip}
          />
        )
      )}

      {legendItems.length > 0 && (
        <div
          style={{
            position: "absolute",
            left: 10,
            bottom: 10,
            maxWidth: "calc(100% - 20px)",
            display: "flex",
            flexWrap: "wrap",
            gap: "6px 10px",
            padding: "6px 8px",
            borderRadius: 8,
            border: "1px solid #d2deee",
            background: "rgba(255, 255, 255, 0.9)",
            boxShadow: "0 3px 10px rgba(20, 32, 51, 0.12)",
            pointerEvents: "none",
          }}
        >
          {legendItems.map((item) => (
            <span
              key={item.category}
              style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--text)" }}
            >
              <i
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 999,
                  background: item.color,
                  border: "1px solid rgba(0, 0, 0, 0.15)",
                }}
              />
              {item.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
