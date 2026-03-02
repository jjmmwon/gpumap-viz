import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

function patchPerformanceMeasureForLargeReactProps() {
  if (!import.meta.env.DEV || typeof performance === 'undefined') return

  type MeasureArgs = Parameters<Performance['measure']>
  const originalMeasure = performance.measure.bind(performance) as (
    ...args: MeasureArgs
  ) => PerformanceMeasure

  performance.measure = ((...args: MeasureArgs) => {
    try {
      return originalMeasure(...args)
    } catch (error) {
      const isDataCloneError =
        error instanceof DOMException && error.name === 'DataCloneError'
      if (!isDataCloneError) throw error

      const [, startOrOptions] = args
      if (startOrOptions && typeof startOrOptions === 'object') {
        const fallback: PerformanceMeasureOptions = {}
        if (startOrOptions.start !== undefined) fallback.start = startOrOptions.start
        if (startOrOptions.end !== undefined) fallback.end = startOrOptions.end
        return originalMeasure(args[0], fallback)
      }
      return originalMeasure(args[0])
    }
  }) as Performance['measure']
}

patchPerformanceMeasureForLargeReactProps()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
