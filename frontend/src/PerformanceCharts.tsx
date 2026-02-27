import type { CSSProperties, ReactNode } from "react";
import type { IterationStat } from "./useGpumap";

interface PerformanceChartsProps {
  history: IterationStat[];
  targetLatency: number;
  phase: "idle" | "connecting" | "running" | "done" | "error";
  nPoints: number;
  totalPoints: number;
}

const CHART_WIDTH = 420;
const CHART_HEIGHT = 88;
const QUEUE_CHART_HEIGHT = CHART_HEIGHT;
const MARGIN = { top: 6, right: 8, bottom: 15, left: 34 };
const PLOT_WIDTH = CHART_WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;
const LEGEND_PAD_LEFT = `${((MARGIN.left / CHART_WIDTH) * 100).toFixed(2)}%`;
const LEGEND_PAD_RIGHT = `${((MARGIN.right / CHART_WIDTH) * 100).toFixed(2)}%`;

function maxValue(values: number[]): number {
  let max = 0;
  for (const value of values) {
    if (value > max) max = value;
  }
  return max;
}

function linePath(values: number[], xAt: (index: number) => number, yAt: (value: number) => number): string {
  if (values.length === 0) return "";
  return values
    .map((value, index) => `${index === 0 ? "M" : "L"}${xAt(index).toFixed(2)} ${yAt(value).toFixed(2)}`)
    .join(" ");
}

function areaPath(
  upper: number[],
  lower: number[],
  xAt: (index: number) => number,
  yAt: (value: number) => number,
): string {
  if (upper.length === 0) return "";
  const top = upper
    .map((value, index) => `${index === 0 ? "M" : "L"}${xAt(index).toFixed(2)} ${yAt(value).toFixed(2)}`)
    .join(" ");
  const bottom = lower
    .map((_, revIndex) => {
      const index = lower.length - 1 - revIndex;
      return `L${xAt(index).toFixed(2)} ${yAt(lower[index]).toFixed(2)}`;
    })
    .join(" ");
  return `${top} ${bottom} Z`;
}

function formatSeconds(value: number): string {
  return value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

function formatOps(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return `${Math.round(value)}`;
}

function ChartEmpty() {
  return <div className="performance-empty">Iteration stat가 아직 없습니다.</div>;
}

function ChartFrame({
  title,
  children,
  legend,
}: {
  title: string;
  children: ReactNode;
  legend: Array<{ label: string; className: string }>;
}) {
  const legendAlignStyle = {
    "--legend-pad-left": LEGEND_PAD_LEFT,
    "--legend-pad-right": LEGEND_PAD_RIGHT,
  } as CSSProperties;

  return (
    <section className="performance-section" style={legendAlignStyle}>
      <h3>{title}</h3>
      <div className="performance-chart-wrap">{children}</div>
      <div className="performance-legend">
        {legend.map((item) => (
          <span key={item.label}>
            <i className={`performance-swatch ${item.className}`} />
            {item.label}
          </span>
        ))}
      </div>
    </section>
  );
}

function chartScales(count: number, maxY: number) {
  const denom = Math.max(1, count - 1);
  const safeMaxY = Math.max(1e-6, maxY);
  const xAt = (index: number) => MARGIN.left + (index / denom) * PLOT_WIDTH;
  const yAt = (value: number) => MARGIN.top + PLOT_HEIGHT - (value / safeMaxY) * PLOT_HEIGHT;
  return { xAt, yAt };
}

function xTickStep(count: number): number {
  if (count <= 12) return 1;
  const targetTickCount = 12;
  const rawStep = Math.max(1, Math.ceil((count - 1) / (targetTickCount - 1)));
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  if (normalized <= 1) return magnitude;
  if (normalized <= 2) return 2 * magnitude;
  if (normalized <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

function xTicks(count: number): number[] {
  if (count <= 0) return [];
  const step = xTickStep(count);
  const ticks: number[] = [];
  for (let value = 1; value <= count; value += step) {
    ticks.push(value);
  }
  if (ticks[ticks.length - 1] !== count) {
    ticks.push(count);
  }
  return ticks;
}

function Axis({
  count,
  maxY,
  labelFormatter,
}: {
  count: number;
  maxY: number;
  labelFormatter: (value: number) => string;
}) {
  const { xAt, yAt } = chartScales(count, maxY);
  const yTicks = [0, maxY / 2, maxY];
  const tickValues = xTicks(count);
  return (
    <>
      {yTicks.map((tick, index) => {
        const y = yAt(tick);
        return (
          <g key={index}>
            <line className="performance-grid" x1={MARGIN.left} y1={y} x2={MARGIN.left + PLOT_WIDTH} y2={y} />
            <text className="performance-axis-label" x={MARGIN.left - 6} y={y + 4} textAnchor="end">
              {labelFormatter(tick)}
            </text>
          </g>
        );
      })}
      <line
        className="performance-axis"
        x1={MARGIN.left}
        y1={MARGIN.top + PLOT_HEIGHT}
        x2={MARGIN.left + PLOT_WIDTH}
        y2={MARGIN.top + PLOT_HEIGHT}
      />
      {tickValues.map((tickValue, index) => {
        const x = xAt(tickValue - 1);
        const isFirst = index === 0;
        const isLast = index === tickValues.length - 1;
        const textAnchor = isFirst ? "start" : isLast ? "end" : "middle";
        return (
          <g key={tickValue}>
            <line
              className="performance-axis-tick"
              x1={x}
              y1={MARGIN.top + PLOT_HEIGHT}
              x2={x}
              y2={MARGIN.top + PLOT_HEIGHT + 4}
            />
            <text className="performance-axis-label" x={x} y={CHART_HEIGHT - 7} textAnchor={textAnchor}>
              {tickValue}
            </text>
          </g>
        );
      })}
    </>
  );
}

function LatencyChart({ history, targetLatency }: { history: IterationStat[]; targetLatency: number }) {
  if (history.length === 0) return <ChartEmpty />;

  const wall = history.map((item) => item.wallTime);
  const kernel = history.map((item) => item.insertionTime + item.updateTime + item.embeddingTime);
  const maxY = Math.max(maxValue(wall), maxValue(kernel), targetLatency);
  const { xAt, yAt } = chartScales(history.length, maxY);

  return (
    <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label="Actual Elapsed Time vs Target">
      <Axis count={history.length} maxY={maxY} labelFormatter={formatSeconds} />
      <line
        className="performance-target"
        x1={MARGIN.left}
        y1={yAt(targetLatency)}
        x2={MARGIN.left + PLOT_WIDTH}
        y2={yAt(targetLatency)}
      />
      <path className="performance-line-wall" d={linePath(wall, xAt, yAt)} />
      <path className="performance-line-kernel" d={linePath(kernel, xAt, yAt)} />
    </svg>
  );
}

function OpsChart({ history }: { history: IterationStat[] }) {
  if (history.length === 0) return <ChartEmpty />;

  const insertionOps = history.map((item) => item.insertionOps);
  const updateOps = history.map((item) => item.updateOps);
  const embeddingOpsInK = history.map((item) => item.embeddingOps / 1000);
  const maxY = Math.max(maxValue(insertionOps), maxValue(updateOps), maxValue(embeddingOpsInK));
  const { xAt, yAt } = chartScales(history.length, maxY);

  return (
    <svg
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      role="img"
      aria-label="Estimated Ops per Iteration (Estimator Learning)"
    >
      <Axis count={history.length} maxY={maxY} labelFormatter={formatOps} />
      <path className="performance-line-insertion" d={linePath(insertionOps, xAt, yAt)} />
      <path className="performance-line-update" d={linePath(updateOps, xAt, yAt)} />
      <path className="performance-line-embedding" d={linePath(embeddingOpsInK, xAt, yAt)} />
    </svg>
  );
}

function KernelBreakdownChart({
  history,
  targetLatency,
}: {
  history: IterationStat[];
  targetLatency: number;
}) {
  if (history.length === 0) return <ChartEmpty />;

  const insertion = history.map((item) => item.insertionTime);
  const update = history.map((item) => item.updateTime);
  const embedding = history.map((item) => item.embeddingTime);
  const cumulativeInsertion = insertion;
  const cumulativeUpdate = insertion.map((value, index) => value + update[index]);
  const cumulativeEmbedding = insertion.map((value, index) => value + update[index] + embedding[index]);
  const zeroLine = insertion.map(() => 0);

  const maxY = Math.max(maxValue(cumulativeEmbedding), targetLatency);
  const { xAt, yAt } = chartScales(history.length, maxY);

  return (
    <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label="Kernel Time Breakdown (Stacked)">
      <Axis count={history.length} maxY={maxY} labelFormatter={formatSeconds} />
      <path className="performance-area-insertion" d={areaPath(cumulativeInsertion, zeroLine, xAt, yAt)} />
      <path className="performance-area-update" d={areaPath(cumulativeUpdate, cumulativeInsertion, xAt, yAt)} />
      <path className="performance-area-embedding" d={areaPath(cumulativeEmbedding, cumulativeUpdate, xAt, yAt)} />
      <line
        className="performance-target"
        x1={MARGIN.left}
        y1={yAt(targetLatency)}
        x2={MARGIN.left + PLOT_WIDTH}
        y2={yAt(targetLatency)}
      />
      <path className="performance-line-total" d={linePath(cumulativeEmbedding, xAt, yAt)} />
    </svg>
  );
}

function QueueDetailChart({ history }: { history: IterationStat[] }) {
  if (history.length === 0) return <ChartEmpty />;

  const latestWithQueue = [...history]
    .reverse()
    .find((item) => item.embeddingQueueLevels.length > 0);
  if (!latestWithQueue) return <ChartEmpty />;

  const levels = latestWithQueue.embeddingQueueLevels;
  const queueMargin = { top: 6, right: 8, bottom: 16, left: 34 };
  const queuePlotWidth = CHART_WIDTH - queueMargin.left - queueMargin.right;
  const queuePlotHeight = QUEUE_CHART_HEIGHT - queueMargin.top - queueMargin.bottom;
  const maxQueue = Math.max(1, maxValue(levels));
  const slotWidth = queuePlotWidth / Math.max(1, levels.length);
  const barWidth = Math.max(3, slotWidth * 0.64);
  const yAt = (value: number) => queueMargin.top + queuePlotHeight - (value / maxQueue) * queuePlotHeight;
  const xAt = (index: number) => queueMargin.left + index * slotWidth + (slotWidth - barWidth) / 2;
  const midTick = Math.floor(maxQueue / 2);
  const yTicks = midTick > 0 ? [0, midTick, maxQueue] : [0, maxQueue];
  const levelStep = xTickStep(levels.length);
  const levelTickIndexes: number[] = [];
  for (let idx = 0; idx < levels.length; idx += levelStep) {
    levelTickIndexes.push(idx);
  }
  if (levels.length > 0 && levelTickIndexes[levelTickIndexes.length - 1] !== levels.length - 1) {
    levelTickIndexes.push(levels.length - 1);
  }

  return (
    <svg viewBox={`0 0 ${CHART_WIDTH} ${QUEUE_CHART_HEIGHT}`} role="img" aria-label="Embedding Queue Detail">
      {yTicks.map((tick, idx) => {
        const y = yAt(tick);
        return (
          <g key={idx}>
            {tick > 0 && (
              <line
                className="performance-grid"
                x1={queueMargin.left}
                y1={y}
                x2={queueMargin.left + queuePlotWidth}
                y2={y}
              />
            )}
            <text className="performance-axis-label" x={queueMargin.left - 6} y={y + 4} textAnchor="end">
              {formatOps(tick)}
            </text>
          </g>
        );
      })}
      {levels.map((value, idx) => {
        const rawBarHeight = (value / maxQueue) * queuePlotHeight;
        const barHeight = value > 0 ? Math.max(1, rawBarHeight) : 0;
        const barY = queueMargin.top + queuePlotHeight - barHeight;
        const x = xAt(idx);
        return (
          <g key={idx}>
            <rect
              className="performance-queue-bar-bg"
              x={x}
              y={queueMargin.top}
              width={barWidth}
              height={queuePlotHeight}
              rx={2}
            />
            <rect
              className="performance-queue-bar"
              x={x}
              y={barY}
              width={barWidth}
              height={barHeight}
              rx={2}
            >
              <title>{`L${idx}: ${formatOps(value)}`}</title>
            </rect>
          </g>
        );
      })}
      <line
        className="performance-axis"
        x1={queueMargin.left}
        y1={queueMargin.top + queuePlotHeight}
        x2={queueMargin.left + queuePlotWidth}
        y2={queueMargin.top + queuePlotHeight}
      />
      {levelTickIndexes.map((idx, tickIndex) => {
        const x = xAt(idx) + barWidth / 2;
        const isFirst = tickIndex === 0;
        const isLast = tickIndex === levelTickIndexes.length - 1;
        const textAnchor = isFirst ? "start" : isLast ? "end" : "middle";
        return (
          <g key={idx}>
            <line
              className="performance-axis-tick"
              x1={x}
              y1={queueMargin.top + queuePlotHeight}
              x2={x}
              y2={queueMargin.top + queuePlotHeight + 4}
            />
            <text className="performance-axis-label" x={x} y={QUEUE_CHART_HEIGHT - 7} textAnchor={textAnchor}>
              {`L${idx}`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function PerformanceCharts({
  history,
  targetLatency,
  phase,
  nPoints,
  totalPoints,
}: PerformanceChartsProps) {
  const progressPct =
    totalPoints > 0 ? Math.min(100, Math.round((nPoints / totalPoints) * 100)) : 0;

  return (
    <section className="progress-card performance-stack">
      <div className="performance-title-wrap">
        <h2 className="panel-title performance-title">Progressive Status View</h2>
      </div>

      <section className="performance-section">
        <h3>Job Queues</h3>
        <p>Insertion: {nPoints.toLocaleString()} / {totalPoints ? totalPoints.toLocaleString() : "-"}</p>
        <p>Phase: {phase}</p>
        <p>Progress: {totalPoints ? `${progressPct}%` : "-"}</p>
      </section>
      <hr className="performance-divider" />

      <ChartFrame
        title="Actual Elapsed Time vs Target"
        legend={[
          { label: "wall time", className: "wall" },
          { label: "kernel time", className: "kernel" },
          { label: `target (${targetLatency}s)`, className: "target" },
        ]}
      >
        <LatencyChart history={history} targetLatency={targetLatency} />
      </ChartFrame>
      <hr className="performance-divider" />

      <ChartFrame
        title="Estimated Ops per Iteration"
        legend={[
          { label: "insertion ops", className: "insertion" },
          { label: "update ops", className: "update" },
          { label: "embedding ops (x1k)", className: "embedding" },
        ]}
      >
        <OpsChart history={history} />
      </ChartFrame>
      <hr className="performance-divider" />

      <ChartFrame
        title="Kernel Time Breakdown"
        legend={[
          { label: "insertion time", className: "area-insertion" },
          { label: "update time", className: "area-update" },
          { label: "embedding time", className: "area-embedding" },
          { label: `target (${targetLatency}s)`, className: "target" },
        ]}
      >
        <KernelBreakdownChart history={history} targetLatency={targetLatency} />
      </ChartFrame>
      <hr className="performance-divider" />

      <ChartFrame
        title="Embedding Queue Detail"
        legend={[
          { label: "remaining jobs per level", className: "queue" },
        ]}
      >
        <QueueDetailChart history={history} />
      </ChartFrame>
    </section>
  );
}
