"use client"

import { useMemo, useState, useCallback, useEffect } from "react"
import { UPlotChart, type Series } from "@/components/charts/UPlotChart"
import { useTheme } from "@/hooks/use-theme"
import { Button } from "@/components/ui/button"
import { Download, Lasso, Filter, Lock, Unlock } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { PairMetrics } from "./metrics"
import { exportPairMetricsToCSV, downloadCSV, filterPairsByThresholds, type FilterThresholds } from "./metrics"
import type { DoublePeakThresholds } from "./types"

type ScaleType = "linear" | "log"

interface DoublePeakScatterProps {
  metrics: PairMetrics
  thresholds: DoublePeakThresholds
  selectedIndices: number[]
  onSelectedIndicesChange: (indices: number[]) => void
  className?: string
}

export function DoublePeakScatter({
  metrics,
  thresholds,
  selectedIndices,
  onSelectedIndicesChange,
  className = "",
}: DoublePeakScatterProps) {
  const { theme } = useTheme()
  const isDark = theme === "dark"
  const [enableLasso, setEnableLasso] = useState(false)
  const [lockZoom, setLockZoom] = useState(false)
  const [yScale, setYScale] = useState<ScaleType>("linear")
  const [xRange, setXRange] = useState<{ min: number; max: number } | null>(null)
  const [filteredIndices, setFilteredIndices] = useState<number[]>([])

  const pointColor = isDark ? "rgba(203, 213, 225, 0.6)" : "rgba(15, 23, 42, 0.6)"
  const thresholdColor = isDark ? "#fb923c" : "#f97316"
  
  // Calculate filtered indices based on thresholds and lasso selection
  useEffect(() => {
    const filtered = filterPairsByThresholds(metrics, thresholds as FilterThresholds, selectedIndices.length > 0 ? selectedIndices : undefined)
    setFilteredIndices(filtered)
  }, [metrics, thresholds, selectedIndices])

  // Create filtered set for quick lookup
  const filteredSet = useMemo(() => new Set(filteredIndices), [filteredIndices])
  const selectedSet = useMemo(() => new Set(selectedIndices), [selectedIndices])

  // Clamp distances for log scale to keep uPlot happy when values hit zero/neg
  const distanceData = useMemo(() => {
    const safeDistances = metrics.distanceMs.map((d) =>
      yScale === "log" ? Math.max(d, 1e-6) : d
    )
    return Float32Array.from(safeDistances)
  }, [metrics.distanceMs, yScale])
  
  // Compute y-range with padding; include thresholds so guide lines stay in view
  const distanceRange = useMemo(() => {
    const values: number[] = []
    for (const v of metrics.distanceMs) {
      if (Number.isFinite(v)) values.push(v)
    }
    thresholds.distance.forEach((v) => {
      if (Number.isFinite(v)) values.push(v)
    })
    if (!values.length) return null

    if (yScale === "log") {
      const positives = values.filter((v) => v > 0)
      if (!positives.length) return null
      let min = Math.min(...positives)
      let max = Math.max(...positives)
      const paddingFactor = 0.15
      min = Math.max(min / (1 + paddingFactor), min * 0.8, 1e-6)
      max = max * (1 + paddingFactor)
      if (max <= min) {
        max = min * 10
      }
      const MAX_RATIO = 1e4
      if (max / min > MAX_RATIO) {
        const mid = Math.sqrt(min * max)
        const half = Math.sqrt(MAX_RATIO)
        min = mid / half
        max = mid * half
      }
      return { min, max }
    }

    const min = Math.min(...values)
    const max = Math.max(...values)
    const span = max - min
    const padding = span > 0 ? span * 0.08 : Math.abs(max) * 0.05 || 1
    return {
      min: Math.max(0, min - padding),
      max: max + padding,
    }
  }, [metrics.distanceMs, thresholds.distance, yScale])

  const scatterSeries: Series[] = useMemo(() => {
    if (metrics.totalPairs === 0) return []

    // All points series with base color - NO LINE CONNECTION
    return [
      {
        label: "Distance between peaks",
        color: pointColor,
        data: distanceData,
        points: true,
        pointSize: 4,
        width: 0, // Explicitly disable line connection between points
      },
    ]
  }, [metrics.totalPairs, pointColor, distanceData])

  const xData = useMemo(
    () => Float32Array.from(metrics.timesMin),
    [metrics.timesMin]
  )

  // Horizontal lines for distance thresholds (clamped for log view)
  const hLines = useMemo(() => {
    const safeMin = yScale === "log" ? Math.max(distanceRange?.min ?? 1e-6, 1e-6) : undefined
    return [
      { y: safeMin ? Math.max(thresholds.distance[0], safeMin) : thresholds.distance[0], color: thresholdColor, dash: [5, 3], label: "Min" },
      { y: safeMin ? Math.max(thresholds.distance[1], safeMin) : thresholds.distance[1], color: thresholdColor, dash: [5, 3], label: "Max" },
    ]
  }, [thresholds.distance, thresholdColor, yScale, distanceRange])

  const handleLassoComplete = useCallback(
    (indices: number[]) => {
      onSelectedIndicesChange(indices)
    },
    [onSelectedIndicesChange]
  )

  const handleExport = useCallback(() => {
    const csv = exportPairMetricsToCSV(metrics, filteredIndices.length > 0 ? filteredIndices : undefined)
    const filename = filteredIndices.length > 0
      ? `double_peak_pairs_filtered_${filteredIndices.length}.csv`
      : `double_peak_pairs_all_${metrics.totalPairs}.csv`
    downloadCSV(csv, filename)
  }, [metrics, filteredIndices])

  const handleClearSelection = useCallback(() => {
    onSelectedIndicesChange([])
  }, [onSelectedIndicesChange])

  if (metrics.totalPairs === 0) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <p className="text-sm text-muted-foreground">No pair data available</p>
      </div>
    )
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Controls */}
      <Card className="mb-4">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">Distance vs Time Scatter</CardTitle>
            <div className="flex flex-wrap items-center gap-2 justify-end">
              <Button
                size="sm"
                variant={enableLasso ? "default" : "outline"}
                onClick={() => setEnableLasso(!enableLasso)}
                className="text-xs"
              >
                <Lasso className="h-3 w-3 mr-1" />
                {enableLasso ? "Lasso Active" : "Enable Lasso"}
              </Button>
              <div className="flex items-center gap-1 pl-1 pr-2 py-1 rounded-md bg-muted/40">
                <span className="text-[11px] text-muted-foreground">Y-scale</span>
                <Button
                  size="sm"
                  variant={yScale === "linear" ? "default" : "outline"}
                  onClick={() => setYScale("linear")}
                  className="text-xs px-2"
                >
                  Linear
                </Button>
                <Button
                  size="sm"
                  variant={yScale === "log" ? "default" : "outline"}
                  onClick={() => setYScale("log")}
                  className="text-xs px-2"
                >
                  Log
                </Button>
              </div>
              <Button
                size="sm"
                variant={lockZoom ? "default" : "outline"}
                onClick={() => setLockZoom(!lockZoom)}
                className="text-xs"
              >
                {lockZoom ? <Lock className="h-3 w-3 mr-1" /> : <Unlock className="h-3 w-3 mr-1" />}
                {lockZoom ? "Zoom Locked" : "Lock Zoom"}
              </Button>
              {selectedIndices.length > 0 && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleClearSelection}
                  className="text-xs"
                >
                  Clear Lasso ({selectedIndices.length})
                </Button>
              )}
              <Button
                size="sm"
                variant="outline"
                onClick={handleExport}
                className="text-xs"
                disabled={filteredIndices.length === 0}
              >
                <Download className="h-3 w-3 mr-1" />
                Export {filteredIndices.length > 0 && filteredIndices.length < metrics.totalPairs ? "Filtered" : "All"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pb-3">
          <div className="text-xs space-y-1">
            <p className="text-muted-foreground">
              Total pairs: <span className="font-medium text-foreground">{metrics.totalPairs}</span>
              {" • "}
              Filtered: <span className={`font-medium ${filteredIndices.length < metrics.totalPairs ? "text-accent" : "text-foreground"}`}>
                {filteredIndices.length}
              </span>
              {selectedIndices.length > 0 && (
                <>
                  {" • "}
                  Lasso: <span className="font-medium text-accent">{selectedIndices.length}</span>
                </>
              )}
            </p>
            <p className="text-muted-foreground text-[11px]">
              {enableLasso
                ? "Click and drag to draw a lasso around points to select them"
                : "Scroll to zoom Y-axis • Shift+scroll to zoom X-axis • Double-click to reset • Lock Zoom to prevent reset"}
            </p>
            {filteredIndices.length < metrics.totalPairs && (
              <p className="text-[11px] text-amber-600 dark:text-amber-500 flex items-center gap-1">
                <Filter className="h-3 w-3" />
                {filteredIndices.length} pairs pass all threshold filters
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Scatter Plot */}
      <div className="flex-1 border rounded-lg p-4 bg-card">
        <UPlotChart
          xData={xData}
          series={scatterSeries}
          xLabel="Time (min)"
          yLabel="Distance (ms)"
          height={500}
          hLines={hLines}
          yScaleType={yScale}
          yRange={distanceRange ?? undefined}
          enableYAxisZoom={true}
          enableLasso={enableLasso}
          onLassoComplete={handleLassoComplete}
          selectedIndices={selectedIndices}
          filteredIndices={filteredIndices}
          customScatter
          xRange={lockZoom ? xRange : null}
          onXRangeChange={setXRange}
        />
      </div>
    </div>
  )
}
