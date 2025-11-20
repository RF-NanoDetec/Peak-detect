"use client"

import { useMemo, useCallback, useState, useEffect, type Dispatch, type SetStateAction } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { UPlotHistogram } from "@/components/charts/UPlotHistogram"
import { Loader2 } from "lucide-react"
import { apiClient } from "@/lib/apiClient"
import type { PairMetrics } from "./metrics"
import type { DoublePeakThresholds } from "./types"

interface DoublePeakPanelProps {
  metrics: PairMetrics
  thresholds: DoublePeakThresholds
  onThresholdsChange: Dispatch<SetStateAction<DoublePeakThresholds>>
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
  className = "",
}: DoublePeakPanelProps) {
  const [histograms, setHistograms] = useState<DoublePeakHistograms | null>(null)
  const [loading, setLoading] = useState(false)

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
          metrics.promOverAmp,
          {
            bin_count: 60,
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
  }, [metrics])

  const updateThreshold = useCallback(
    (key: keyof DoublePeakThresholds, index: 0 | 1, value: number) => {
      onThresholdsChange(prev => {
        const nextRange = [...prev[key]] as [number, number]
        nextRange[index] = value
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
      { value: thresholds.distance[0], color: "#ef4444", label: "Min" },
      { value: thresholds.distance[1], color: "#ef4444", label: "Max" },
    ],
    [thresholds.distance[0], thresholds.distance[1]]
  )

  const pairPromRatioLines = useMemo(
    () => [
      { value: thresholds.pairPromRatio[0], color: "#ef4444", label: "Min" },
      { value: thresholds.pairPromRatio[1], color: "#ef4444", label: "Max" },
    ],
    [thresholds.pairPromRatio[0], thresholds.pairPromRatio[1]]
  )

  const pairWidthRatioLines = useMemo(
    () => [
      { value: thresholds.pairWidthRatio[0], color: "#ef4444", label: "Min" },
      { value: thresholds.pairWidthRatio[1], color: "#ef4444", label: "Max" },
    ],
    [thresholds.pairWidthRatio[0], thresholds.pairWidthRatio[1]]
  )

  const promOverAmpLines = useMemo(
    () => [
      { value: thresholds.promOverAmp[0], color: "#ef4444", label: "Min" },
      { value: thresholds.promOverAmp[1], color: "#ef4444", label: "Max" },
    ],
    [thresholds.promOverAmp[0], thresholds.promOverAmp[1]]
  )

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Distance Constraint */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Distance (ms)</CardTitle>
          <CardDescription>Distance between consecutive peaks</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min (ms)</label>
              <Input
                type="number"
                value={thresholds.distance[0]}
                onChange={(e) => updateThreshold("distance", 0, parseFloat(e.target.value) || 0)}
                step="0.1"
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max (ms)</label>
              <Input
                type="number"
                value={thresholds.distance[1]}
                onChange={(e) => updateThreshold("distance", 1, parseFloat(e.target.value) || 0)}
                step="1"
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
              xLabel="Distance (ms)"
              yLabel="Count"
              xScaleType="log"
              height={180}
              verticalLines={distanceLines}
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
          <CardTitle className="text-base">Pair Prominence Ratio</CardTitle>
          <CardDescription>Prom[i+1] / Prom[i] for consecutive peaks</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min</label>
              <Input
                type="number"
                value={thresholds.pairPromRatio[0]}
                onChange={(e) => updateThreshold("pairPromRatio", 0, parseFloat(e.target.value) || 0)}
                step="0.1"
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max</label>
              <Input
                type="number"
                value={thresholds.pairPromRatio[1]}
                onChange={(e) => updateThreshold("pairPromRatio", 1, parseFloat(e.target.value) || 0)}
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
              verticalLines={pairPromRatioLines}
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
          <CardTitle className="text-base">Pair Width Ratio</CardTitle>
          <CardDescription>Width[i+1] / Width[i] for consecutive peaks</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min</label>
              <Input
                type="number"
                value={thresholds.pairWidthRatio[0]}
                onChange={(e) => updateThreshold("pairWidthRatio", 0, parseFloat(e.target.value) || 0)}
                step="0.1"
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max</label>
              <Input
                type="number"
                value={thresholds.pairWidthRatio[1]}
                onChange={(e) => updateThreshold("pairWidthRatio", 1, parseFloat(e.target.value) || 0)}
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
              verticalLines={pairWidthRatioLines}
            />
          ) : (
            <div className="flex items-center justify-center h-[180px] border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">No data</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Per-Peak Prominence/Amplitude Ratio */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Prominence / Amplitude</CardTitle>
          <CardDescription>Per-peak prominence-to-amplitude ratio</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium">Min</label>
              <Input
                type="number"
                value={thresholds.promOverAmp[0]}
                onChange={(e) => updateThreshold("promOverAmp", 0, parseFloat(e.target.value) || 0)}
                step="0.01"
                className="text-xs"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max</label>
              <Input
                type="number"
                value={thresholds.promOverAmp[1]}
                onChange={(e) => updateThreshold("promOverAmp", 1, parseFloat(e.target.value) || 0)}
                step="0.01"
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
              xLabel="Prom / Amp"
              yLabel="Count"
              xScaleType="linear"
              height={180}
              verticalLines={promOverAmpLines}
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

