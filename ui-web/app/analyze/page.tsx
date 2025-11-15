"use client"

import { BarChart3 } from "lucide-react"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { AnalyzeTimeSeries } from "@/components/charts/AnalyzeTimeSeries"

export default function AnalyzePage() {
  const router = useRouter()
  const { detectionResults } = useResultsStore()
  const { previewData } = useDataStore()
  const { params } = useParamsStore()

  const hasResults = !!(detectionResults && previewData)

  return (
    <PageShell>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Analyze</h2>
            <p className="text-sm text-muted-foreground">
              Time series and distributions of detected peaks
            </p>
          </div>
        </div>
      </PageControls>

      <PageVisualization>
        {hasResults ? (
          <div className="flex-1 p-4 space-y-4 overflow-auto">
            <div>
              <p className="text-xs text-muted-foreground">
                Scatter plots of detected peak amplitudes, widths, and throughput over time.
                All time axes are linked �?� scroll to zoom, double-click to reset.
              </p>
            </div>
            <AnalyzeTimeSeries
              className="flex-1"
              peakTimes={detectionResults?.peak_times || []}
              peakAmplitudes={detectionResults?.peak_amplitudes || []}
              peakIntervalsMs={detectionResults?.peak_intervals || []}
              peakWidths={detectionResults?.properties?.widths || []}
              timeResolution={params.time_resolution}
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
              <Button onClick={() => router.push('/detect')}>
                Detect Peaks
              </Button>
            </div>
          </div>
        )}
      </PageVisualization>
    </PageShell>
  )
}
