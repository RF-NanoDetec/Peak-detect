"use client"

import { useMemo, useCallback, useState, useEffect, type Dispatch, type SetStateAction } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { UnitStepperInput } from "@/components/ui/unit-input"
import { UPlotHistogram } from "@/components/charts/UPlotHistogram"
import { Loader2 } from "lucide-react"
import { apiClient } from "@/lib/apiClient"
import type { PairMetrics } from "./metrics"
import type { DoublePeakThresholds } from "./types"
import { InfoTooltip } from "@/components/ui/info-tooltip"
import { useTheme } from "@/hooks/use-theme"

interface DoublePeakPanelProps {
  metrics: PairMetrics
  thresholds: DoublePeakThresholds
  onThresholdsChange: Dispatch<SetStateAction<DoublePeakThresholds>>
  resolutionMs: number
  filteredIndices: number[]
  filteredMetrics: PairMetrics
  className?: string
}

type HistogramData = {
  bins: number[]
  counts: number[]
  bin_edges?: number[]
}

interface DoublePeakHistograms {
  distance: HistogramData
  pairPromRatio: HistogramData
  pairWidthRatio: HistogramData
  promOverAmp: HistogramData
}

export function DoublePeakPanel({
  metrics,
  thresholds,
  onThresholdsChange,
  resolutionMs,
  filteredIndices,
  filteredMetrics,
  className = "",
}: DoublePeakPanelProps) {
  const [histograms, setHistograms] = useState<DoublePeakHistograms | null>(null)
  const [loading, setLoading] = useState(false)
  const [binCount, setBinCount] = useState(100)
  const { theme } = useTheme()
  const isDark = theme === "dark"
  
  // Accent color for threshold lines
  const thresholdColor = isDark ? "#fb923c" : "#f97316"
  const baseHistogramColor = isDark ? "#334155" : "#1f2937"

  // Fetch histograms when metrics change
  useEffect(() => {
    if (metrics.totalPairs === 0) {
      setHistograms(null)
      return
    }

    const fetchHistograms = async () => {
      setLoading(true)
      try {
        const data = await apiClient.generateDoubleHistograms(
          metrics.distanceMs,
          metrics.pairPromRatio,
          metrics.pairWidthRatio,
          metrics.promOverAmp.map((val) => val * 100), // percent view for prominence/amplitude
          {
            bin_count: binCount,
            metrics: {
              distance: { xScale: "log", yScale: "linear" },
              pairPromRatio: { xScale: "linear", yScale: "linear" },
              pairWidthRatio: { xScale: "linear", yScale: "linear" },
              promOverAmp: { xScale: "linear", yScale: "linear" },
            },
          }
        )
        setHistograms(data)
      } catch (error) {
        console.error("Failed to fetch histograms:", error)
        setHistograms(null)
      } finally {
        setLoading(false)
      }
    }

    fetchHistograms()
  }, [metrics, binCount])

  const updateThreshold = useCallback(
    (
      key: keyof DoublePeakThresholds,
      index: 0 | 1,
      value: number,
      opts?: { min?: number; max?: number }
    ) => {
      if (!Number.isFinite(value)) return
      const min = opts?.min ?? -Infinity
      const max = opts?.max ?? Infinity
      const clamped = Math.min(Math.max(value, min), max)
      onThresholdsChange((prev) => {
        const nextRange = [...prev[key]] as [number, number]
        nextRange[index] = clamped
        if (index === 0 && nextRange[1] < clamped) {
          nextRange[1] = clamped
        } else if (index === 1 && nextRange[0] > clamped) {
          nextRange[0] = clamped
        }
        return {
          ...prev,
          [key]: nextRange,
        }
      })
    },
    [onThresholdsChange]
  )

  const distanceLines = useMemo(
    () => [
      { value: thresholds.distance[0], color: thresholdColor, label: "Min" },
      { value: thresholds.distance[1], color: thresholdColor, label: "Max" },
    ],
    [thresholds.distance, thresholdColor]
  )

  const pairPromRatioLines = useMemo(
    () => [
      { value: thresholds.pairPromRatio[0], color: thresholdColor, label: "Min" },
      { value: thresholds.pairPromRatio[1], color: thresholdColor, label: "Max" },
    ],
    [thresholds.pairPromRatio, thresholdColor]
  )

  const pairWidthRatioLines = useMemo(
    () => [
      { value: thresholds.pairWidthRatio[0], color: thresholdColor, label: "Min" },
      { value: thresholds.pairWidthRatio[1], color: thresholdColor, label: "Max" },
    ],
    [thresholds.pairWidthRatio, thresholdColor]
  )

  const promOverAmpLines = useMemo(
    () => [
      { value: thresholds.promOverAmp[0], color: thresholdColor, label: "Min" },
      { value: thresholds.promOverAmp[1], color: thresholdColor, label: "Max" },
    ],
    [thresholds.promOverAmp, thresholdColor]
  )

  const deriveEdges = useCallback(
    (base: HistogramData | null, scale: "linear" | "log") => {
      if (!base) return null
      if (base.bin_edges && base.bin_edges.length > 1) {
        return base.bin_edges
      }
      const bins = base.bins || []
      if (bins.length < 2) return null

      if (scale === "log") {
        const ratio = bins[1] > 0 ? bins[1] / bins[0] : NaN
        if (!Number.isFinite(ratio) || ratio <= 0) return null
        const start = bins[0] / Math.sqrt(ratio)
        const edges: number[] = [start]
        for (let i = 0; i < bins.length; i++) {
          edges.push(edges[i] * ratio)
        }
        return edges
      }

      const step = bins[1] - bins[0]
      if (!Number.isFinite(step) || step === 0) return null
      const start = bins[0] - step / 2
      const edges: number[] = []
      for (let i = 0; i <= bins.length; i++) {
        edges.push(start + step * i)
      }
      return edges
    },
    []
  )

  const buildOverlayHistogram = useCallback(
    (base: HistogramData | null, values: number[], scale: "linear" | "log"): HistogramData | null => {
      if (!base || !base.bins || base.bins.length === 0 || values.length === 0) return null
      const edges = deriveEdges(base, scale)
      if (!edges || edges.length < 2) return null

      const counts = new Array(edges.length - 1).fill(0)
      const minEdge = edges[0]
      const maxEdge = edges[edges.length - 1]

      for (let i = 0; i < values.length; i++) {
        const v = values[i]
        if (!Number.isFinite(v)) continue
        if (scale === "log" && v <= 0) continue
        if (v < minEdge || v > maxEdge) continue

        let left = 0
        let right = edges.length - 2
        while (left <= right) {
          const mid = (left + right) >> 1
          const low = edges[mid]
          const high = edges[mid + 1]
          if (v < low) {
            right = mid - 1
          } else if (v > high) {
            left = mid + 1
          } else {
            counts[mid] += 1
            break
          }
        }
      }

      return {
        bins: base.bins,
        counts,
        bin_edges: edges,
      }
    },
    [deriveEdges]
  )

  const overlayHistograms = useMemo(() => {
    if (!histograms || filteredIndices.length === 0 || filteredMetrics.totalPairs === 0) return null

    return {
      distance: buildOverlayHistogram(histograms.distance, filteredMetrics.distanceMs, "log"),
      pairPromRatio: buildOverlayHistogram(histograms.pairPromRatio, filteredMetrics.pairPromRatio, "linear"),
      pairWidthRatio: buildOverlayHistogram(histograms.pairWidthRatio, filteredMetrics.pairWidthRatio, "linear"),
      promOverAmp: buildOverlayHistogram(
        histograms.promOverAmp,
        filteredMetrics.promOverAmp.map((val) => val * 100),
        "linear"
      ),
    }
  }, [histograms, filteredMetrics, filteredIndices.length, buildOverlayHistogram])

  return (
    <div className={`space-y-4 ${className}`}>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            Histogram Bin Count
            <InfoTooltip content="Adjust the number of bins used for all histograms. Updates apply immediately." />
          </CardTitle>
          <CardDescription>Refine histogram resolution</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium">
            <span className="text-muted-foreground">Bins</span>
            <span className="px-2 py-1 rounded bg-muted text-foreground/80">{binCount}</span>
          </div>
          <input
            type="range"
            min={20}
            max={200}
            step={10}
            value={binCount}
            onChange={(e) => setBinCount(parseInt(e.target.value, 10))}
            className="w-full"
          />
          <div className="flex justify-between text-[10px] text-muted-foreground px-1">
            <span>Fewer bins</span>
            <span>More bins</span>
          </div>
        </CardContent>
      </Card>

      {/* Distance Constraint */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Distance
            <InfoTooltip content="Time separation between two peaks in a potential pair. Peaks too close or too far apart are not considered pairs." />
          </CardTitle>
          <CardDescription>Distance between consecutive peaks</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min</label>
              <UnitStepperInput
                unit="ms"
                value={thresholds.distance[0]}
                onValueChange={(value) => updateThreshold("distance", 0, value, { min: resolutionMs })}
                step={0.1}
                min={resolutionMs}
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max</label>
              <UnitStepperInput
                unit="ms"
                value={thresholds.distance[1]}
                onValueChange={(value) => updateThreshold("distance", 1, value, { min: thresholds.distance[0] })}
                step={1}
                min={thresholds.distance[0]}
                className="text-xs"
              />
            </div>
          </div>
          {loading ? (
            <div className="flex items-center justify-center h-[180px]">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : histograms && histograms.distance ? (
            <UPlotHistogram
              data={histograms.distance}
              xLabel="Distance [ms]"
              yLabel="Count"
              xScaleType="log"
              height={180}
              color={baseHistogramColor}
              overlayData={overlayHistograms?.distance ?? undefined}
              overlayColor={thresholdColor}
              verticalLines={distanceLines}
            />
          ) : (
            <div className="flex items-center justify-center h-[180px] border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">No data</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Per-Peak Prominence Ratio (percent view) */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Prominence Ratio
            <InfoTooltip content="Per-peak prominence divided by amplitude, expressed as a percentage. Matches the Prominence Ratio setting used during detection." />
          </CardTitle>
          <CardDescription>Prominence / Amplitude, shown as %</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min</label>
              <UnitStepperInput
                unit="%"
                value={thresholds.promOverAmp[0]}
                onValueChange={(value) => updateThreshold("promOverAmp", 0, value, { min: 0, max: 200 })}
                step={1}
                min={0}
                max={200}
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max</label>
              <UnitStepperInput
                unit="%"
                value={thresholds.promOverAmp[1]}
                onValueChange={(value) => updateThreshold("promOverAmp", 1, value, { min: 0, max: 200 })}
                step={1}
                min={0}
                max={200}
                className="text-xs"
              />
            </div>
          </div>
          {loading ? (
            <div className="flex items-center justify-center h-[180px]">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : histograms && histograms.promOverAmp ? (
            <UPlotHistogram
              data={histograms.promOverAmp}
              xLabel="Prominence Ratio (%)"
              yLabel="Count"
              xScaleType="linear"
              height={180}
              color={baseHistogramColor}
              overlayData={overlayHistograms?.promOverAmp ?? undefined}
              overlayColor={thresholdColor}
              verticalLines={promOverAmpLines}
            />
          ) : (
            <div className="flex items-center justify-center h-[180px] border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">No data</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pair Width Ratio */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Pair Width Ratio
            <InfoTooltip content="Ratio of the second peak's width to the first peak's width. Ensures the pair components have comparable shapes." />
          </CardTitle>
          <CardDescription>Width[i+1] / Width[i] for consecutive peaks</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min</label>
              <Input
                type="number"
                value={thresholds.pairWidthRatio[0]}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val)) {
                    updateThreshold("pairWidthRatio", 0, val, { min: 0 })
                  }
                }}
                step="0.1"
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max</label>
              <Input
                type="number"
                value={thresholds.pairWidthRatio[1]}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val)) {
                    updateThreshold("pairWidthRatio", 1, val, { min: 0 })
                  }
                }}
                step="0.1"
                className="text-xs"
              />
            </div>
          </div>
          {loading ? (
            <div className="flex items-center justify-center h-[180px]">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : histograms && histograms.pairWidthRatio ? (
            <UPlotHistogram
              data={histograms.pairWidthRatio}
              xLabel="Width Ratio"
              yLabel="Count"
              xScaleType="linear"
              height={180}
              color={baseHistogramColor}
              overlayData={overlayHistograms?.pairWidthRatio ?? undefined}
              overlayColor={thresholdColor}
              verticalLines={pairWidthRatioLines}
            />
          ) : (
            <div className="flex items-center justify-center h-[180px] border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">No data</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pair Prominence Ratio */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Pair Prominence Ratio
            <InfoTooltip content="Ratio of the second peak's prominence to the first peak's prominence. Used to identify pairs with similar (or specific dissimilar) intensities." />
          </CardTitle>
          <CardDescription>Prom[i+1] / Prom[i] for consecutive peaks</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min</label>
              <Input
                type="number"
                value={thresholds.pairPromRatio[0]}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val)) {
                    updateThreshold("pairPromRatio", 0, val, { min: 0 })
                  }
                }}
                step="0.1"
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max</label>
              <Input
                type="number"
                value={thresholds.pairPromRatio[1]}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val)) {
                    updateThreshold("pairPromRatio", 1, val, { min: 0 })
                  }
                }}
                step="0.1"
                className="text-xs"
              />
            </div>
          </div>
          {loading ? (
            <div className="flex items-center justify-center h-[180px]">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : histograms && histograms.pairPromRatio ? (
            <UPlotHistogram
              data={histograms.pairPromRatio}
              xLabel="Prominence Ratio"
              yLabel="Count"
              xScaleType="linear"
              height={180}
              color={baseHistogramColor}
              overlayData={overlayHistograms?.pairPromRatio ?? undefined}
              overlayColor={thresholdColor}
              verticalLines={pairPromRatioLines}
            />
          ) : (
            <div className="flex items-center justify-center h-[180px] border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">No data</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
