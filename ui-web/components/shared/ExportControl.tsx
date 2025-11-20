"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Download } from "lucide-react"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"

interface ExportControlProps {
  hasDoublePeakAnalysis?: boolean
  doublePeakAnalysis?: any
  doublePeakParams?: {
    min_distance: number
    max_distance: number
    min_amp_ratio: number
    max_amp_ratio: number
    min_width_ratio: number
    max_width_ratio: number
  }
}

export function ExportControl({ hasDoublePeakAnalysis, doublePeakAnalysis, doublePeakParams }: ExportControlProps) {
  const [format, setFormat] = useState<"csv" | "txt" | "xlsx">("csv")
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [filterDoublePeaks, setFilterDoublePeaks] = useState(false)
  const [isExporting, setIsExporting] = useState(false)

  const { detectionResults } = useResultsStore()
  const { resultId, filteredResultId } = useDataStore()
  const { params } = useParamsStore()

  const handleExport = async () => {
    if (!resultId) return

    setIsExporting(true)
    try {
      const metadata: Record<string, any> = {
        "Protocol": "Standard", // TODO: Get actual protocol if available
        "Time Resolution": params.time_resolution,
        "Filter Type": params.filter_type,
        "Filter Cutoff": params.filter_cutoff_freq,
        "Peak Threshold": params.prominence_threshold,
        "Peak Distance": params.distance,
        "Peak Width Range": params.width_ms,
        "Prominence Ratio": params.prominence_ratio,
        "Rel Height": params.rel_height,
      }
      
      if (hasDoublePeakAnalysis && doublePeakParams) {
        metadata["Double Peak Min Distance"] = doublePeakParams.min_distance
        metadata["Double Peak Max Distance"] = doublePeakParams.max_distance
        metadata["Double Peak Min Amp Ratio"] = doublePeakParams.min_amp_ratio
        metadata["Double Peak Max Amp Ratio"] = doublePeakParams.max_amp_ratio
        metadata["Double Peak Min Width Ratio"] = doublePeakParams.min_width_ratio
        metadata["Double Peak Max Width Ratio"] = doublePeakParams.max_width_ratio
      }

      const payload = {
        resultId,
        filteredResultId,
        double_peak_analysis: doublePeakAnalysis,
        double_peak_params: doublePeakParams,
        metadata,
        format,
        include_metadata: includeMetadata,
        filter_double_peaks: filterDoublePeaks,
        // Pass detection params to ensure consistency if re-detection needed
        prominence_threshold: params.prominence_threshold,
        distance: params.distance,
        rel_height: params.rel_height,
        width_ms: params.width_ms,
        prominence_ratio: params.prominence_ratio,
        time_resolution: params.time_resolution,
        // Pass existing peaks to avoid re-detection if possible
        peaks: detectionResults?.peaks,
        properties: detectionResults?.properties
      }

      const response = await fetch("/api/export/unified", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) throw new Error("Export failed")

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `peaks_analysis.${format === 'xlsx' ? 'xlsx' : format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error("Export error:", error)
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Download className="h-4 w-4" />
          Export Data
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Format</label>
          <div className="flex gap-2">
            {(["csv", "txt", "xlsx"] as const).map((f) => (
              <Button
                key={f}
                variant={format === f ? "default" : "outline"}
                size="sm"
                onClick={() => setFormat(f)}
                className="flex-1 uppercase"
              >
                {f}
              </Button>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Include Metadata</label>
          <Switch checked={includeMetadata} onCheckedChange={setIncludeMetadata} />
        </div>

        {hasDoublePeakAnalysis && (
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Filter Double Peaks Only</label>
            <Switch checked={filterDoublePeaks} onCheckedChange={setFilterDoublePeaks} />
          </div>
        )}

        <Button 
          className="w-full" 
          onClick={handleExport} 
          disabled={isExporting || !resultId}
        >
          {isExporting ? "Exporting..." : "Download"}
        </Button>
      </CardContent>
    </Card>
  )
}
