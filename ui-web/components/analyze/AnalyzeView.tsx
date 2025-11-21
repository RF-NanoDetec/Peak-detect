import { BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { EmptyState } from "@/components/ui/empty-state"
import { AnalyzeTimeSeries } from "@/components/charts/AnalyzeTimeSeries"
import type { DetectionResults } from "@/lib/stores/resultsStore" // Or wherever types are

interface AnalyzeViewProps {
  hasResults: boolean
  detectionResults: DetectionResults | null
  params: any
  binWidthSeconds: number
  rollingMeanWindow: number
  onDetectPeaks: () => void
}

export function AnalyzeView({
  hasResults,
  detectionResults,
  params,
  binWidthSeconds,
  rollingMeanWindow,
  onDetectPeaks,
}: AnalyzeViewProps) {
  if (!hasResults) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <EmptyState
          icon={BarChart3}
          title="No Analysis Results"
          description="Complete peak detection to view analysis results."
          action={
            <Button onClick={onDetectPeaks}>
              Detect Peaks
            </Button>
          }
        />
      </div>
    )
  }

  return (
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
  )
}

