"use client"

import { useState, useMemo, useCallback, useEffect } from "react"
import { Download, Loader2, CheckCircle, FileSpreadsheet, AlertCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Separator } from "@/components/ui/separator"
import { InfoTooltip } from "@/components/ui/info-tooltip"
import { toast } from "sonner"
import { apiClient } from "@/lib/apiClient"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useDoubleStore } from "@/lib/stores/doubleStore"
import { useRouter } from "next/navigation"
import { computePairMetrics, filterPairsByThresholds } from "@/components/double/metrics"
import type { DoublePeakThresholds } from "@/components/double/types"
import type { PairMetrics } from "@/components/double/metrics"

type ExportMode = "single" | "double" | "combined"

export default function ExportPage() {
  const router = useRouter()
  const { resultId, filteredResultId } = useDataStore()
  const { params } = useParamsStore()
  const { detectionResults } = useResultsStore()
  const { selectedPairIndices } = useDoubleStore()
  
  const [isExporting, setIsExporting] = useState(false)
  const [format, setFormat] = useState<"csv" | "txt" | "xlsx">("csv")
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [exportMode, setExportMode] = useState<ExportMode>("single")
  
  // Double peak thresholds (same defaults as Double Peak page)
  const [thresholds, setThresholds] = useState<DoublePeakThresholds>({
    distance: [0.1, 100.0] as [number, number],
    pairPromRatio: [0.01, 100.0] as [number, number],
    pairWidthRatio: [0.01, 100.0] as [number, number],
    promOverAmp: [0.0, 1.0] as [number, number],
  })

  // Compute pair metrics from detection results (same as Double Peak page)
  const pairMetrics: PairMetrics = useMemo(() => {
    if (!detectionResults || !detectionResults.peak_times || detectionResults.peak_times.length === 0) {
      return {
        distanceMs: [],
        pairPromRatio: [],
        pairWidthRatio: [],
        timesMin: [],
        promOverAmp: [],
        totalPairs: 0,
        totalPeaks: 0,
      }
    }

    const peakTimes = detectionResults.peak_times || []
    const peakAmplitudes = detectionResults.peak_amplitudes || []
    const prominences = detectionResults.properties?.prominences || []
    const widths = detectionResults.properties?.widths || []
    const timeResolution = params.time_resolution || 1e-4

    if (peakTimes.length === 0 || prominences.length === 0) {
      return {
        distanceMs: [],
        pairPromRatio: [],
        pairWidthRatio: [],
        timesMin: [],
        promOverAmp: [],
        totalPairs: 0,
        totalPeaks: 0,
      }
    }

    return computePairMetrics({
      peakTimes,
      peakAmplitudes,
      prominences,
      widths,
      timeResolution,
    })
  }, [detectionResults, params.time_resolution])

  // Count filtered pairs based on thresholds
  const filteredPairIndices = useMemo(() => {
    if (pairMetrics.totalPairs === 0) return []
    return filterPairsByThresholds(pairMetrics, thresholds)
  }, [pairMetrics, thresholds])

  const clampedSelection = useMemo(
    () => selectedPairIndices.filter((idx) => idx >= 0 && idx < pairMetrics.totalPairs),
    [pairMetrics.totalPairs, selectedPairIndices]
  )

  const activePairIndices = useMemo(() => {
    if (clampedSelection.length > 0) {
      return clampedSelection
    }
    return filteredPairIndices
  }, [clampedSelection, filteredPairIndices])

  const selectionSource = clampedSelection.length > 0 ? "lasso" : "filters"

  const hasData = !!resultId
  const hasDetectionResults = !!(detectionResults && detectionResults.peak_times && detectionResults.peak_times.length > 0)
  const hasPairs = pairMetrics.totalPairs > 0

  useEffect(() => {
    if (!hasPairs && exportMode !== "single") {
      setExportMode("single")
    }
  }, [exportMode, hasPairs])

  const updateThreshold = useCallback(
    (key: keyof DoublePeakThresholds, index: 0 | 1, value: number) => {
      setThresholds(prev => {
        const nextRange = [...prev[key]] as [number, number]
        nextRange[index] = value
        return {
          ...prev,
          [key]: nextRange,
        }
      })
    },
    []
  )

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  const handleExport = async () => {
    if (!resultId) {
      toast.error("No data to export")
      return
    }

    if (!hasDetectionResults) {
      toast.error("Please run peak detection first")
      return
    }

    setIsExporting(true)
    try {
      const metadata: Record<string, any> = {
        "Protocol": "Standard", 
        "Time Resolution": params.time_resolution,
        "Filter Type": params.filter_type,
        "Filter Cutoff": params.filter_cutoff_freq,
        "Peak Threshold": params.prominence_threshold,
        "Peak Distance": params.distance,
        "Peak Width Range": params.width_ms,
        "Prominence Ratio": params.prominence_ratio,
        "Rel Height": params.rel_height,
      }

      // Double peak params for the API (only when exporting double data)
      const doublePeakParams = {
        min_distance: thresholds.distance[0] / 1000, // Convert ms to seconds
        max_distance: thresholds.distance[1] / 1000,
        min_amp_ratio: thresholds.pairPromRatio[0],
        max_amp_ratio: thresholds.pairPromRatio[1],
        min_width_ratio: thresholds.pairWidthRatio[0],
        max_width_ratio: thresholds.pairWidthRatio[1],
      }

      if (exportMode !== "single" && hasPairs) {
        metadata["Double Peak Min Distance (ms)"] = thresholds.distance[0]
        metadata["Double Peak Max Distance (ms)"] = thresholds.distance[1]
        metadata["Double Peak Min Prom Ratio"] = thresholds.pairPromRatio[0]
        metadata["Double Peak Max Prom Ratio"] = thresholds.pairPromRatio[1]
        metadata["Double Peak Min Width Ratio"] = thresholds.pairWidthRatio[0]
        metadata["Double Peak Max Width Ratio"] = thresholds.pairWidthRatio[1]
        metadata["Double Peak Selected Pairs"] = activePairIndices.length
        metadata["Double Peak Selection Source"] = selectionSource
      }

      const payload = {
        resultId,
        filteredResultId,
        double_peak_params: exportMode === "single" ? undefined : doublePeakParams,
        double_peak_indices: exportMode === "single" ? undefined : activePairIndices,
        double_peak_thresholds: exportMode === "single" ? undefined : thresholds,
        metadata,
        format,
        include_metadata: includeMetadata,
        filter_double_peaks: exportMode === "double",
        export_mode: exportMode,
        selection_source: selectionSource,
        prominence_threshold: params.prominence_threshold,
        distance: params.distance,
        rel_height: params.rel_height,
        width_ms: params.width_ms,
        prominence_ratio: params.prominence_ratio,
        time_resolution: params.time_resolution,
        peaks: detectionResults?.peaks,
        properties: detectionResults?.properties
      }

      const blob = await apiClient.exportUnified(payload)
      
      const filenameBase =
        exportMode === "single"
          ? "single_peaks_analysis"
          : exportMode === "double"
            ? "double_peaks_analysis"
            : "combined_peaks_analysis"

      const filename = `${filenameBase}.${format === 'xlsx' ? 'xlsx' : format}`
      
      downloadBlob(blob, filename)
      toast.success(`Exported successfully as ${format.toUpperCase()}`)
    } catch (error) {
      console.error("Export error:", error)
      toast.error("Failed to export data")
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <PageShell>
      <PageVisualization>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center space-y-6 max-w-lg">
            <div className="mx-auto w-20 h-20 rounded-full bg-accent/10 flex items-center justify-center">
              <FileSpreadsheet className="h-10 w-10 text-accent" />
            </div>
            
            <div>
              <h3 className="text-xl font-semibold">Export Your Data</h3>
              <p className="text-sm text-muted-foreground mt-2">
                Export peak detection results and double peak analysis data. 
                Configure your export options in the panel on the right.
              </p>
            </div>

            {/* Export Summary */}
            {hasDetectionResults && (
              <Card className="text-left">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Export Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Total Peaks Detected</span>
                      <span className="font-medium">{detectionResults?.peak_times?.length || 0}</span>
                    </div>
                    {hasPairs && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Peak Pairs</span>
                          <span className="font-medium">{pairMetrics.totalPairs}</span>
                        </div>
                        {exportMode !== "single" && (
                          <div className="flex justify-between text-accent">
                            <span>Pairs to Export</span>
                            <span className="font-semibold">
                              {activePairIndices.length}
                              {selectionSource === "lasso" ? " (lasso)" : " (filters)"}
                            </span>
                          </div>
                        )}
                      </>
                    )}
                    <Separator />
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Export Format</span>
                      <span className="font-medium uppercase">{format}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Include Metadata</span>
                      <span className="font-medium">{includeMetadata ? "Yes" : "No"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Export Mode</span>
                      <span className="font-medium">
                        {exportMode === "single" ? "Single Peaks" : exportMode === "double" ? "Double Peaks" : "Combined"}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {!hasDetectionResults && hasData && (
              <div className="flex items-center gap-2 text-amber-600 bg-amber-50 dark:bg-amber-900/20 px-4 py-3 rounded-lg">
                <AlertCircle className="h-5 w-5" />
                <span className="text-sm">Run peak detection to enable export</span>
              </div>
            )}

            {!hasData && (
              <div className="flex items-center gap-2 text-muted-foreground bg-muted/50 px-4 py-3 rounded-lg">
                <AlertCircle className="h-5 w-5" />
                <span className="text-sm">Load data to get started</span>
              </div>
            )}
          </div>
        </div>
      </PageVisualization>

      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Export</h2>
            <p className="text-sm text-muted-foreground">
              Export analysis results
            </p>
          </div>

          {/* Status Card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Data Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Data Loaded</span>
                  <span className={hasData ? "text-accent flex items-center gap-1" : "text-muted-foreground/70"}>
                    {hasData ? <><CheckCircle className="h-3.5 w-3.5" /> Yes</> : "✗ No"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Preprocessing Applied</span>
                  <span className={filteredResultId ? "text-accent flex items-center gap-1" : "text-muted-foreground/50"}>
                    {filteredResultId ? <><CheckCircle className="h-3.5 w-3.5" /> Yes</> : "○ Optional"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Peaks Detected</span>
                  <span className={hasDetectionResults ? "text-accent flex items-center gap-1" : "text-muted-foreground/70"}>
                    {hasDetectionResults ? <><CheckCircle className="h-3.5 w-3.5" /> {detectionResults?.peak_times?.length} peaks</> : "✗ Not yet"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Peak Pairs Available</span>
                  <span className={hasPairs ? "text-accent flex items-center gap-1" : "text-muted-foreground/50"}>
                    {hasPairs ? <><CheckCircle className="h-3.5 w-3.5" /> {pairMetrics.totalPairs} pairs</> : "○ Need 2+ peaks"}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {!hasDetectionResults ? (
            <div className="text-center py-6 space-y-3">
              <p className="text-sm text-muted-foreground">
                {hasData 
                  ? "Run peak detection to enable data export"
                  : "Load data and run peak detection to export"}
              </p>
              <Button 
                size="sm" 
                variant="outline"
                onClick={() => router.push(hasData ? '/preprocess' : '/load')}
              >
                {hasData ? "Go to Process & Detect" : "Go to Load Data"}
              </Button>
            </div>
          ) : (
            <Accordion type="multiple" defaultValue={["format", "options"]} className="w-full">
              {/* Format Selection */}
              <AccordionItem value="format" className="border-none">
                <AccordionTrigger className="py-2 hover:no-underline">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    Export Format
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pt-2 pb-4 space-y-3">
                  <div className="flex gap-2">
                    {(["csv", "txt", "xlsx"] as const).map((f) => (
                      <Button
                        key={f}
                        variant={format === f ? "default" : "outline"}
                        size="sm"
                        onClick={() => setFormat(f)}
                        className="flex-1 uppercase text-xs h-9"
                      >
                        {f}
                      </Button>
                    ))}
                  </div>
                </AccordionContent>
              </AccordionItem>

              <Separator className="my-1" />

              {/* Export Options */}
              <AccordionItem value="options" className="border-none">
                <AccordionTrigger className="py-2 hover:no-underline">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    Export Options
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pt-2 pb-4 space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium flex items-center gap-1">
                      Include Metadata
                      <InfoTooltip content="Include analysis parameters and settings in the export header" />
                    </label>
                    <Switch checked={includeMetadata} onCheckedChange={setIncludeMetadata} />
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-medium flex items-center gap-1">
                      Export Mode
                      <InfoTooltip content="Choose whether to export single peaks, double-peak pairs, or both together." />
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {(["single", "double", "combined"] as ExportMode[]).map((mode) => (
                        <Button
                          key={mode}
                          variant={exportMode === mode ? "default" : "outline"}
                          size="sm"
                          className="text-xs h-9"
                          onClick={() => setExportMode(mode)}
                          disabled={mode !== "single" && !hasPairs}
                        >
                          {mode === "single" ? "Single" : mode === "double" ? "Double" : "Combined"}
                        </Button>
                      ))}
                    </div>
                    {clampedSelection.length > 0 && (
                      <p className="text-[11px] text-accent">
                        Using lasso selection ({clampedSelection.length} pairs)
                      </p>
                    )}
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Double Peak Filters - only show when enabled */}
              {hasPairs && exportMode !== "single" && (
                <>
                  <Separator className="my-1" />
                  
                  <AccordionItem value="filters" className="border-none">
                    <AccordionTrigger className="py-2 hover:no-underline">
                      <div className="flex items-center gap-2 text-sm font-medium">
                        Double Peak Filters
                        <span className="text-xs font-normal text-muted-foreground ml-1">
                          ({activePairIndices.length} of {pairMetrics.totalPairs})
                        </span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pt-2 pb-4 space-y-4">
                      {clampedSelection.length > 0 && (
                        <p className="text-[11px] text-muted-foreground">
                          Lasso selection is active; clear it in the Double Peak page to use filters.
                        </p>
                      )}
                      {/* Distance */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium flex items-center gap-1">
                          Distance (ms)
                          <InfoTooltip content="Time separation between consecutive peaks in a pair" />
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Min</span>
                            <Input
                              type="number"
                              value={thresholds.distance[0]}
                              onChange={(e) => updateThreshold("distance", 0, parseFloat(e.target.value) || 0)}
                              step="0.1"
                              className="h-8 text-xs"
                            />
                          </div>
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Max</span>
                            <Input
                              type="number"
                              value={thresholds.distance[1]}
                              onChange={(e) => updateThreshold("distance", 1, parseFloat(e.target.value) || 0)}
                              step="1"
                              className="h-8 text-xs"
                            />
                          </div>
                        </div>
                      </div>

                      {/* Prominence Ratio */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium flex items-center gap-1">
                          Prominence Ratio
                          <InfoTooltip content="Ratio of second peak's prominence to first peak's prominence" />
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Min</span>
                            <Input
                              type="number"
                              value={thresholds.pairPromRatio[0]}
                              onChange={(e) => updateThreshold("pairPromRatio", 0, parseFloat(e.target.value) || 0)}
                              step="0.1"
                              className="h-8 text-xs"
                            />
                          </div>
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Max</span>
                            <Input
                              type="number"
                              value={thresholds.pairPromRatio[1]}
                              onChange={(e) => updateThreshold("pairPromRatio", 1, parseFloat(e.target.value) || 0)}
                              step="0.1"
                              className="h-8 text-xs"
                            />
                          </div>
                        </div>
                      </div>

                      {/* Width Ratio */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium flex items-center gap-1">
                          Width Ratio
                          <InfoTooltip content="Ratio of second peak's width to first peak's width" />
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Min</span>
                            <Input
                              type="number"
                              value={thresholds.pairWidthRatio[0]}
                              onChange={(e) => updateThreshold("pairWidthRatio", 0, parseFloat(e.target.value) || 0)}
                              step="0.1"
                              className="h-8 text-xs"
                            />
                          </div>
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Max</span>
                            <Input
                              type="number"
                              value={thresholds.pairWidthRatio[1]}
                              onChange={(e) => updateThreshold("pairWidthRatio", 1, parseFloat(e.target.value) || 0)}
                              step="0.1"
                              className="h-8 text-xs"
                            />
                          </div>
                        </div>
                      </div>

                      {/* Prom/Amp Ratio */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium flex items-center gap-1">
                          Prom / Amplitude
                          <InfoTooltip content="Per-peak prominence-to-amplitude ratio filter" />
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Min</span>
                            <Input
                              type="number"
                              value={thresholds.promOverAmp[0]}
                              onChange={(e) => updateThreshold("promOverAmp", 0, parseFloat(e.target.value) || 0)}
                              step="0.01"
                              className="h-8 text-xs"
                            />
                          </div>
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground">Max</span>
                            <Input
                              type="number"
                              value={thresholds.promOverAmp[1]}
                              onChange={(e) => updateThreshold("promOverAmp", 1, parseFloat(e.target.value) || 0)}
                              step="0.01"
                              className="h-8 text-xs"
                            />
                          </div>
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </>
              )}
            </Accordion>
          )}

          {/* Export Button */}
          {hasDetectionResults && (
            <Button 
              className="w-full mt-4" 
              onClick={handleExport} 
              disabled={
                isExporting ||
                !hasDetectionResults ||
                (exportMode !== "single" && (!hasPairs || activePairIndices.length === 0))
              }
              size="lg"
            >
              {isExporting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="h-4 w-4 mr-2" />
                  Download {format.toUpperCase()}
                </>
              )}
            </Button>
          )}
        </div>
      </PageControls>
    </PageShell>
  )
}
