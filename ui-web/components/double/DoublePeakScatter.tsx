"use client"

import { useMemo, useState, useCallback, useEffect, useRef } from "react"
import { UPlotChart, type Series } from "@/components/charts/UPlotChart"
import { useTheme } from "@/hooks/use-theme"
import { Button } from "@/components/ui/button"
import { Download, Lasso, Filter } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { PairMetrics } from "./metrics"
import { exportPairMetricsToCSV, downloadCSV, filterPairsByThresholds, type FilterThresholds } from "./metrics"
import type { DoublePeakThresholds } from "./types"

type ScaleType = "linear" | "log"
type Range = { min: number; max: number } | null

const LOG_FLOOR = 1e-6
const MAX_LOG_RATIO = 1e4
const rangesEqual = (a: Range, b: Range, eps = 1e-6) =>
  !!a &&
  !!b &&
  Math.abs(a.min - b.min) < eps &&
  Math.abs(a.max - b.max) < eps

const safeMin = (arr: number[]) => {
  let min = Infinity
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]
    if (Number.isFinite(v) && v < min) min = v
  }
  return min
}

const safeMax = (arr: number[]) => {
  let max = -Infinity
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]
    if (Number.isFinite(v) && v > max) max = v
  }
  return max
}

const buildRange = (values: number[], scale: ScaleType): Range => {
  if (!values.length) return null
  const filtered: number[] = []
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    if (scale === "log" && v <= 0) continue
    filtered.push(v)
  }
  if (!filtered.length) return null

  let min = safeMin(filtered)
  let max = safeMax(filtered)

  if (!Number.isFinite(min) || !Number.isFinite(max)) return null

  if (scale === "log") {
    const paddingFactor = 0.15
    min = Math.max(min / (1 + paddingFactor), min * 0.8, LOG_FLOOR)
    max = max * (1 + paddingFactor)
    if (max <= min) {
      max = min * 10
    }
    const ratio = max / min
    if (ratio > MAX_LOG_RATIO) {
      const geom = Math.sqrt(min * max)
      const half = Math.sqrt(MAX_LOG_RATIO)
      min = Math.max(LOG_FLOOR, geom / half)
      max = geom * half
    }
    return { min, max }
  }

  const span = max - min
  const padding = span > 0 ? span * 0.08 : Math.abs(max) * 0.05 || 1
  return {
    min: Math.max(0, min - padding),
    max: max + padding,
  }
}

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
  const [yScale, setYScale] = useState<ScaleType>("linear")
  const [xRange, setXRange] = useState<{ min: number; max: number } | null>(null)
  const [filteredIndices, setFilteredIndices] = useState<number[]>([])
  const [yRange, setYRange] = useState<Range>(() => {
    const seedRange = buildRange(
      [...metrics.distanceMs, thresholds.distance[0], thresholds.distance[1]],
      "linear"
    )
    if (seedRange) return seedRange
    const minSeed = Math.max(1, thresholds.distance[0] || 1)
    const maxSeed = Math.max(100, thresholds.distance[1] || 100)
    return { min: minSeed, max: maxSeed }
  })
  const prevDistanceRef = useRef<[number, number] | null>(null)

  const pointColor = isDark ? "rgba(203, 213, 225, 0.6)" : "rgba(15, 23, 42, 0.6)"
  const thresholdColor = isDark ? "#fb923c" : "#f97316"
  const normalizedThresholds: FilterThresholds = useMemo(
    () => ({
      ...thresholds,
      promOverAmp: [
        thresholds.promOverAmp[0] / 100,
        thresholds.promOverAmp[1] / 100,
      ],
    }),
    [thresholds]
  )
  
  // Calculate filtered indices based on thresholds and lasso selection
  useEffect(() => {
    const filtered = filterPairsByThresholds(
      metrics,
      normalizedThresholds,
      selectedIndices.length > 0 ? selectedIndices : undefined
    )
    setFilteredIndices(filtered)
  }, [metrics, normalizedThresholds, selectedIndices])

  // Create filtered set for quick lookup
  const filteredSet = useMemo(() => new Set(filteredIndices), [filteredIndices])
  const selectedSet = useMemo(() => new Set(selectedIndices), [selectedIndices])

  // Clamp distances for log scale to keep uPlot happy when values hit zero/neg
  const distanceData = useMemo(() => {
    const safeDistances = metrics.distanceMs.map((d) =>
      yScale === "log" ? Math.max(d, LOG_FLOOR) : d
    )
    return Float32Array.from(safeDistances)
  }, [metrics.distanceMs, yScale])
  
  // Apply threshold/scale-driven y-range updates (do not override user zoom unless values/scale change)
  useEffect(() => {
    const prev = prevDistanceRef.current
    const changed =
      !prev ||
      prev[0] !== thresholds.distance[0] ||
      prev[1] !== thresholds.distance[1]
    prevDistanceRef.current = [...thresholds.distance]

    const nextRange = buildRange(
      [...metrics.distanceMs, thresholds.distance[0], thresholds.distance[1]],
      yScale
    )

    if (!nextRange) return
    if (changed) {
      setYRange(nextRange)
      return
    }
    // If the current yRange is unsafe for the new scale, snap once
    setYRange((prevRange) => (rangesEqual(prevRange, nextRange) ? prevRange : nextRange))
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
    const floor = yScale === "log" ? Math.max(yRange?.min ?? LOG_FLOOR, LOG_FLOOR) : undefined
    return [
      { y: floor ? Math.max(thresholds.distance[0], floor) : thresholds.distance[0], color: thresholdColor, dash: [5, 3], label: "Min" },
      { y: floor ? Math.max(thresholds.distance[1], floor) : thresholds.distance[1], color: thresholdColor, dash: [5, 3], label: "Max" },
    ]
  }, [thresholds.distance, thresholdColor, yScale, yRange])

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

  const handleYRangeChange = useCallback(
    (range: { min: number; max: number }) => {
      setYRange((prev) => (rangesEqual(prev, range) ? prev : range))
    },
    []
  )

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
                : "Scroll to zoom Y-axis • Shift+scroll to zoom X-axis • Double-click to reset"}
            </p>
            {filteredIndices.length < metrics.totalPairs && (
              <p className="text-[11px] text-amber-600 dark:text-amber-500 flex items-center gap-1">
                <Filter className="h-3 w-3" />
                {filteredIndices.length} pairs pass all threshold filters
              </p>
            )}
          </div>
          <div className="flex items-center justify-end gap-2 mt-2">
            <div className="flex items-center gap-1 pl-2 pr-2 py-1 rounded-md bg-muted/40">
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
          yRange={yRange ?? undefined}
          enableYAxisZoom={true}
          enableLasso={enableLasso}
          onLassoComplete={handleLassoComplete}
          selectedIndices={selectedIndices}
          filteredIndices={filteredIndices}
          customScatter
          xRange={xRange ?? undefined}
          onXRangeChange={setXRange}
          onYRangeChange={handleYRangeChange}
        />
      </div>
    </div>
  )
}
