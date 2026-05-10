import { Button } from "@/components/ui/button"
import { EmptyState } from "@/components/ui/empty-state"
import { AnalyzeTimeSeries } from "@/components/charts/AnalyzeTimeSeries"
import type { DetectPeaksResponse, Parameters } from "@/lib/types"
import { workflowSteps } from "@/components/layout/workflow"

const analyzeStep = workflowSteps.find((step) => step.id === "analyze")!
const preprocessStep = workflowSteps.find((step) => step.id === "preprocess")!

interface AnalyzeViewProps {
  hasResults: boolean
  detectionResults: DetectPeaksResponse | null
  params: Parameters
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
          icon={analyzeStep.icon}
          title="No Analysis Results"
          description="Run peak detection to inspect peak prominence, width, and throughput before exporting."
          action={
            <Button onClick={onDetectPeaks}>
              Go to {preprocessStep.label}
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

