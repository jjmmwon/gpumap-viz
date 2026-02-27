import { useRef, useState } from "react";
import type { GpumapConfig } from "./useGpumap";

interface Props {
  onStart: (config: GpumapConfig) => void;
  onStop: () => void;
  onUploadNpy: (file: File) => Promise<{ shape: number[]; dtype: string } | null>;
  running: boolean;
}

const DEFAULT: GpumapConfig = {
  data_source: "mnist",
  n_neighbors: 15,
  min_dist: 0.1,
  max_epoch: 200,
  target_latency: 10.0,
  verbose: false,
};

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: "var(--muted)", fontSize: 14 }}>{label}</span>
        <span style={{ fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function NumberInputRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ color: "var(--muted)", fontSize: 14, display: "block", marginBottom: 6 }}>
        {label}
      </label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const next = e.currentTarget.valueAsNumber;
          if (!Number.isNaN(next)) onChange(next);
        }}
        style={{
          width: "100%",
          background: "var(--surface-alt)",
          color: "var(--text)",
          border: "1px solid var(--border)",
          borderRadius: 6,
          padding: "6px 8px",
          fontSize: 15,
        }}
      />
    </div>
  );
}

export function ConfigPanel({ onStart, onStop, onUploadNpy, running }: Props) {
  const [cfg, setCfg] = useState<GpumapConfig>(DEFAULT);
  const [npyInfo, setNpyInfo] = useState<{ name: string; shape: number[] } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const set = <K extends keyof GpumapConfig>(k: K, v: GpumapConfig[K]) =>
    setCfg((prev) => ({ ...prev, [k]: v }));

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const info = await onUploadNpy(file);
    if (info) setNpyInfo({ name: file.name, shape: info.shape });
  };

  return (
    <aside className="config-panel">
      <div className="config-panel-title-wrap">
        <h2 className="panel-title">Configuration</h2>
      </div>

      <div className="config-panel-body">
        {/* Data source */}
        <section>
          <label style={{ color: "var(--muted)", fontSize: 14, display: "block", marginBottom: 8 }}>
            DATA SOURCE
          </label>
          <select
            value={cfg.data_source}
            onChange={(e) => set("data_source", e.target.value as GpumapConfig["data_source"])}
            style={{
              width: "100%",
              marginBottom: 8,
              background: "var(--surface-alt)",
              color: "var(--text)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "7px 8px",
              fontSize: 15,
            }}
          >
            <option value="mnist">MNIST</option>
            <option value="npy">NPY</option>
          </select>

          {cfg.data_source === "npy" && (
            <div>
              <button
                onClick={() => fileRef.current?.click()}
                style={{
                  width: "100%",
                  background: "var(--surface-alt)",
                  border: "1px dashed var(--border)",
                  color: "var(--muted)",
                }}
              >
                {npyInfo ? `${npyInfo.name} [${npyInfo.shape.join("×")}]` : "Upload .npy file"}
              </button>
              <input ref={fileRef} type="file" accept=".npy" onChange={handleFile} />
            </div>
          )}
        </section>

        <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "12px 0" }} />

        <NumberInputRow
          label="n_neighbors"
          value={cfg.n_neighbors}
          min={5}
          max={32}
          step={1}
          onChange={(v) => set("n_neighbors", v)}
        />
        <NumberInputRow
          label="min_dist"
          value={cfg.min_dist}
          min={0.0}
          max={1.0}
          step={0.01}
          onChange={(v) => set("min_dist", v)}
        />
        <NumberInputRow
          label="max_epoch"
          value={cfg.max_epoch}
          min={100}
          max={500}
          step={50}
          onChange={(v) => set("max_epoch", v)}
        />
        <SliderRow
          label="target_latency (s)"
          value={cfg.target_latency}
          min={0.1} max={20.0} step={0.1}
          format={(v) => v.toFixed(2)}
          onChange={(v) => set("target_latency", v)}
        />

        <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "12px 0" }} />

        {/* Verbose */}
        <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={cfg.verbose}
            onChange={(e) => set("verbose", e.target.checked)}
            style={{ accentColor: "var(--accent)" }}
          />
          <span style={{ fontSize: 15, color: "var(--muted)" }}>verbose logging</span>
        </label>

        {/* <div style={{ flex: 1 }} /> */}

        {/* Start / Stop */}
        <button
          onClick={() => (running ? onStop() : onStart(cfg))}
          disabled={cfg.data_source === "npy" && !npyInfo}
          style={{
            width: "100%",
            background: running ? "var(--danger)" : "var(--accent)",
            borderColor: running ? "var(--danger)" : "var(--accent)",
            color: "#fff",
            marginTop: 12,
            padding: "10px 0",
            fontSize: 17,
          }}
        >
          {running ? "STOP" : "START"}
        </button>
      </div>
    </aside>
  );
}
