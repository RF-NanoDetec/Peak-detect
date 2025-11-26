import { Filter } from "lucide-react"
import { Button } from "@/components/ui/button"
import { EmptyState } from "@/components/ui/empty-state"
import { PreprocessingChart } from "@/components/charts/PreprocessingChart"
import { DetectPeaksResponse, DataPreviewResponse } from "@/lib/types"

interface PreprocessViewProps {
  resultId: string | null
  filteredResultId: string | null
  params: any
  detectionResults: DetectPeaksResponse | null
  previewData: DataPreviewResponse | null
  onLoadData: () => void
}

export function PreprocessView({
  resultId,
  filteredResultId,
  params,
  detectionResults,
  previewData,
  onLoadData,
}: PreprocessViewProps) {
  if (!resultId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <EmptyState
          icon={Filter}
          title="No Data to Preprocess"
          description="Load data first to apply preprocessing filters and detect peaks."
          action={
            <Button onClick={onLoadData}>
              Load Data
            </Button>
          }
        />
      </div>
    )
  }

  return (
    <PreprocessingChart
      resultId={resultId}
      filteredResultId={filteredResultId}
      className="flex-1"
      timeResolution={params.time_resolution}
      peakTimes={detectionResults?.peak_times || []}
      peakAmplitudes={detectionResults?.peak_amplitudes || []}
      peakIntervals={detectionResults?.peak_intervals || []}
      peakProperties={detectionResults?.properties || null}
      previewData={previewData}
      prominenceThreshold={params.prominence_threshold}
      distance={params.distance}
      widthMs={params.width_ms}
      initialHistograms={detectionResults?.histograms}
    />
  )
}

