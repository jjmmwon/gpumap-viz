import type { GpumapStatus } from "./useGpumap";

interface Props {
  status: GpumapStatus;
}

const PHASE_COLOR: Record<string, string> = {
  idle: "var(--muted)",
  connecting: "var(--warning)",
  running: "var(--success)",
  done: "var(--accent)",
  error: "var(--danger)",
};

function Dot({ phase }: { phase: string }) {
  const color = PHASE_COLOR[phase] ?? "var(--muted)";
  const pulse = phase === "running" || phase === "connecting";
  return (
    <span style={{
      display: "inline-block",
      width: 8,
      height: 8,
      borderRadius: "50%",
      background: color,
      flexShrink: 0,
      animation: pulse ? "pulse 1.2s infinite" : undefined,
    }} />
  );
}

export function StatusBar({ status }: Props) {
  const { phase, nInstances, nFeatures, embedding, message } = status;
  const pct = nInstances && embedding
    ? Math.min(100, Math.round((embedding.nPoints / nInstances) * 100))
    : 0;

  return (
    <>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
      <div style={{
        padding: "12px 16px",
        background: "var(--surface-alt)",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        gap: 9,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <Dot phase={phase} />
          <span style={{
            textTransform: "capitalize",
            fontWeight: 600,
            fontSize: 14,
            padding: "2px 8px",
            border: "1px solid var(--border)",
            borderRadius: 999,
            background: "var(--surface)",
          }}>
            {phase}
          </span>
          {message && (
            <span style={{ color: "var(--danger)", fontSize: 14, marginLeft: 2 }}>
              {message}
            </span>
          )}
        </div>

        {embedding && (
          <div style={{ display: "flex", gap: 14, fontSize: 14, color: "var(--muted)", flexWrap: "wrap" }}>
            <span>Points: <b style={{ color: "var(--text)" }}>{embedding.nPoints.toLocaleString()}</b></span>
            {nInstances && (
              <span>Total: <b style={{ color: "var(--text)" }}>{nInstances.toLocaleString()}</b></span>
            )}
            {nFeatures && (
              <span>Dims: <b style={{ color: "var(--text)" }}>{nFeatures}</b></span>
            )}
            {embedding.isDone && (
              <span style={{ color: "var(--success)", fontWeight: 700 }}>Complete</span>
            )}
          </div>
        )}

        {nInstances && pct > 0 && (
          <div style={{
            height: 5,
            background: "#d8e2f0",
            borderRadius: 999,
            overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              width: `${pct}%`,
              background: "var(--accent)",
              transition: "width 0.3s ease",
            }} />
          </div>
        )}
      </div>
    </>
  );
}
