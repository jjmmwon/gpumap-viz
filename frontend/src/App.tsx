import { ConfigPanel } from "./ConfigPanel";
import { PerformanceCharts } from "./PerformanceCharts";
import { ScatterPlot } from "./ScatterPlot";
import { StatusBar } from "./StatusBar";
import { useGpumap } from "./useGpumap";

export default function App() {
  const { status, start, stop, uploadNpy } = useGpumap();
  const running = status.phase === "running" || status.phase === "connecting";
  const nPoints = status.embedding?.nPoints ?? 0;
  const totalPoints = status.nInstances ?? 0;

  return (
    <div className="app-shell">
      <header className="app-global-header">
        <h1 className="app-global-title">
          <span>GPU</span>MAP Viz
        </h1>
        {/* <span className="app-global-phase">{status.phase}</span> */}
      </header>

      <div className="app-body">
        <aside className="app-sidebar">
          <section className="app-config-stage">
            <ConfigPanel
              onStart={start}
              onStop={stop}
              onUploadNpy={uploadNpy}
              running={running}
            />
          </section>

          <section className="app-status-stage">
            <StatusBar status={status} />
          </section>
        </aside>

        <main className="app-main">
          <div className="app-workspace">
            <section className="app-embedding-stage">
              <div className="app-panel-head">
                <h2 className="panel-title">Embedding</h2>
                {/* <p>Embedding View</p> */}
              </div>
              <div className="app-embedding-square">
                <ScatterPlot
                  embedding={status.embedding}
                  classCategories={status.classCategories}
                  classNames={status.classNames}
                />
              </div>
            </section>

            <aside className="app-progress-rail">
              <PerformanceCharts
                history={status.history}
                targetLatency={status.targetLatency}
                phase={status.phase}
                nPoints={nPoints}
                totalPoints={totalPoints}
              />
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}
