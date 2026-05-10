import { Button } from "@/components/ui/button"
import { EmptyState } from "@/components/ui/empty-state"
import { PreprocessingChart } from "@/components/charts/PreprocessingChart"
import type { DetectPeaksResponse, Parameters } from "@/lib/types"
import { workflowSteps } from "@/components/layout/workflow"

const loadStep = workflowSteps.find((step) => step.id === "load")!
const preprocessStep = workflowSteps.find((step) => step.id === "preprocess")!

interface PreprocessViewProps {
  resultId: string | null
  filteredResultId: string | null
  params: Parameters
  detectionResults: DetectPeaksResponse | null
  onLoadData: () => void
}

export function PreprocessView({
  resultId,
  filteredResultId,
  params,
  detectionResults,
  onLoadData,
}: PreprocessViewProps) {
  if (!resultId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <EmptyState
          icon={preprocessStep.icon}
          title="No Data to Preprocess"
          description="Load a supported file first, then inspect the signal and detect peaks."
          action={
            <Button onClick={onLoadData}>
              {loadStep.label}
            </Button>
          }
        />
      </div>
    )
  }

  const peakCount = detectionResults?.peak_times?.length || 0

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="shrink-0 border-b bg-card/60 px-4 py-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">Signal inspection and peak detection</h2>
            <p className="text-xs text-muted-foreground">
              Tune parameters in the control panel, then verify peaks on the chart before export.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="rounded-md border bg-muted/40 px-2.5 py-1">
              {filteredResultId ? "Filtered signal" : "Raw signal"}
            </span>
            <span className="rounded-md border bg-muted/40 px-2.5 py-1">
              {peakCount > 0 ? `${peakCount.toLocaleString()} peaks detected` : "Detection pending"}
            </span>
          </div>
        </div>
      </div>
      <PreprocessingChart
        resultId={resultId}
        filteredResultId={filteredResultId}
        className="flex-1"
        timeResolution={params.time_resolution}
        peakTimes={detectionResults?.peak_times || []}
        peakAmplitudes={detectionResults?.peak_amplitudes || []}
        peakIntervals={detectionResults?.peak_intervals || []}
        peakProperties={detectionResults?.properties || null}
        prominenceThreshold={params.prominence_threshold}
        distance={params.distance}
        widthMs={params.width_ms}
        initialHistograms={detectionResults?.histograms}
      />
    </div>
  )
}

