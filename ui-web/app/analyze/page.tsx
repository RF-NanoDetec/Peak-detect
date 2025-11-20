"use client"

import { useState } from "react"
import { BarChart3 } from "lucide-react"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { AnalyzeTimeSeries } from "@/components/charts/AnalyzeTimeSeries"
import { ExportControl } from "@/components/shared/ExportControl"

export default function AnalyzePage() {
  const router = useRouter()
  const { detectionResults } = useResultsStore()
  const { previewData } = useDataStore()
  const { params } = useParamsStore()

  const hasResults = !!(detectionResults && previewData)
  const [binWidthSeconds, setBinWidthSeconds] = useState<number>(10)
  const [rollingMeanWindow, setRollingMeanWindow] = useState<number>(10)

  return (
    <PageShell>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Analyze & Export</h2>
            <p className="text-sm text-muted-foreground">
              Time series analysis and data export
            </p>
          </div>

          {hasResults && (
            <>
              <div className="space-y-2 text-xs">
                <p className="font-medium">Throughput Binning</p>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Bin width</span>
                    <span className="font-semibold">{binWidthSeconds.toFixed(1)} s</span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={20}
                    step={0.5}
                    value={binWidthSeconds}
                    onChange={(e) => setBinWidthSeconds(parseFloat(e.target.value))}
                    className="w-full accent-primary h-2 rounded-lg cursor-pointer"
                  />
                </div>
              </div>
              <div className="space-y-2 text-xs">
                <p className="font-medium">Rolling Mean</p>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Window size</span>
                    <span className="font-semibold">{rollingMeanWindow}</span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={200}
                    step={1}
                    value={rollingMeanWindow}
                    onChange={(e) => setRollingMeanWindow(Number(e.target.value))}
                    className="w-full accent-primary h-2 rounded-lg cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>2</span>
                    <span>200</span>
                  </div>
                </div>
              </div>
              <ExportControl />
            </>
          )}
        </div>
      </PageControls>

      <PageVisualization>
        {hasResults ? (
          <div className="flex-1 p-4 space-y-4 overflow-auto">
            <AnalyzeTimeSeries
              className="flex-1"
              peakTimes={detectionResults?.peak_times || []}
              peakAmplitudes={detectionResults?.peak_amplitudes || []}
              peakIntervalsMs={detectionResults?.peak_intervals || []}
              peakWidths={detectionResults?.properties?.widths || []}
              timeResolution={params.time_resolution}
              binWidthSeconds={binWidthSeconds}
              rollingMeanWindow={rollingMeanWindow}
            />
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center space-y-4 max-w-md">
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <BarChart3 className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">No Analysis Results</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Complete peak detection to view analysis results
                </p>
              </div>
              <Button onClick={() => router.push('/preprocess')}>
                Detect Peaks
              </Button>
            </div>
          </div>
        )}
      </PageVisualization>
    </PageShell>
  )
}
