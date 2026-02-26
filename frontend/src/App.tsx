import { ConfigPanel } from "./ConfigPanel";
import { ScatterPlot } from "./ScatterPlot";
import { StatusBar } from "./StatusBar";
import { useGpumap } from "./useGpumap";

export default function App() {
  const { status, start, stop, uploadNpy } = useGpumap();
  const running = status.phase === "running" || status.phase === "connecting";
  const nPoints = status.embedding?.nPoints ?? 0;
  const totalPoints = status.nInstances ?? 0;
  const progressPct =
    totalPoints > 0 ? Math.min(100, Math.round((nPoints / totalPoints) * 100)) : 0;

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="app-brand">
          <h1 className="app-title">
            <span>GPU</span>MAP Viz
          </h1>
          {/* <p className="app-subtitle">Embedding Atlas style workspace</p> */}
        </div>

        <ConfigPanel
          onStart={start}
          onStop={stop}
          onUploadNpy={uploadNpy}
          running={running}
        />

        <StatusBar status={status} />
      </aside>

      <main className="app-main">
        <div className="app-main-header">
          <span>Embedding View</span>
          <span>{status.phase}</span>
        </div>

        <div className="app-workspace">
          <section className="app-embedding-stage">
            <div className="app-embedding-square">
              <ScatterPlot embedding={status.embedding} />
            </div>
          </section>

          <aside className="app-progress-rail">
            <section className="progress-card">
              <h3>Job Queues</h3>
              <p>Insertion: {nPoints.toLocaleString()} / {totalPoints ? totalPoints.toLocaleString() : "-"}</p>
              <p>Phase: {status.phase}</p>
              <p>Progress: {totalPoints ? `${progressPct}%` : "-"}</p>
            </section>

            <section className="progress-card">
              <h3>Progress Placeholder</h3>
              <p>실행 지표를 실시간 차트로 붙일 수 있는 슬롯입니다.</p>
              <div className="progress-bar-track">
                <div style={{ width: `${progressPct}%` }} />
              </div>
            </section>

            <section className="progress-card">
              <h3>Iteration Trends</h3>
              <div className="progress-mini-grid">
                {["Insertion", "Update", "Embedding"].map((label) => (
                  <div key={label} className="progress-mini">
                    <span>{label}</span>
                    <div className="progress-mini-line" />
                  </div>
                ))}
              </div>
            </section>

            <section className="progress-card">
              <h3>Queue Detail</h3>
              <p>TODO: 단계별 작업량/지연 시간 시각화 컴포넌트 연결</p>
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}
