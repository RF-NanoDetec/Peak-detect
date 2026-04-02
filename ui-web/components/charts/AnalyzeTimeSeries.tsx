"use client"

import { useMemo, useState, useCallback, useRef } from "react"
import { UPlotChart, type Series } from "./UPlotChart"
import { useTheme } from "@/hooks/use-theme"

interface AnalyzeTimeSeriesProps {
  peakTimes?: number[]
  peakAmplitudes?: number[]
  peakIntervalsMs?: number[]
  peakWidths?: number[]
  timeResolution?: number
  className?: string
  binWidthSeconds?: number
  rollingMeanWindow?: number
}

// Compute dynamic opacity based on point count
// More points = lower opacity to prevent visual clutter
const computePointOpacity = (pointCount: number, isDark: boolean): number => {
  if (pointCount <= 500) {
    return isDark ? 0.5 : 0.6
  } else if (pointCount <= 2000) {
    // Gradual decrease from 0.5/0.6 to 0.15/0.25
    const t = (pointCount - 500) / 1500
    return isDark ? 0.5 - t * 0.35 : 0.6 - t * 0.35
  } else if (pointCount <= 10000) {
    // Continue decreasing for very large datasets
    const t = (pointCount - 2000) / 8000
    return isDark ? 0.15 - t * 0.1 : 0.25 - t * 0.15
  }
  // Minimum opacity for extremely large datasets
  return isDark ? 0.05 : 0.1
}

type ScaleType = "linear" | "log"

type Range = { min: number; max: number } | null

const axisButtonClass = (active: boolean) =>
  `px-2.5 py-1 rounded-md border text-[10px] font-medium transition-colors ${
    active
      ? "bg-accent border-accent text-accent-foreground shadow-sm"
      : "border-border/50 text-muted-foreground hover:text-foreground hover:border-accent/60 hover:bg-accent/10"
  }`

const computeRange = (values: number[], scale: ScaleType, linearMin?: number): Range => {
  // Early exit for empty arrays
  if (!values || values.length === 0) return null
  
  // Use a single pass to find min/max of finite values
  let minVal = Infinity
  let maxVal = -Infinity
  let hasFinite = false
  
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (Number.isFinite(v)) {
      hasFinite = true
      if (v < minVal) minVal = v
      if (v > maxVal) maxVal = v
    }
  }
  
  if (!hasFinite) return null
  
  if (scale === "log") {
    // For log scale, find min/max of positive values only
    let posMin = Infinity
    let posMax = -Infinity
    let hasPositive = false
    
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      if (Number.isFinite(v) && v > 0) {
        hasPositive = true
        if (v < posMin) posMin = v
        if (v > posMax) posMax = v
      }
    }
    
    if (!hasPositive) return null
    
    const paddingFactor = 0.1
    const minPadded = posMin / (1 + paddingFactor)
    const maxPadded = posMax * (1 + paddingFactor)
    return {
      min: Math.max(minPadded, posMin * 0.5),
      max: maxPadded,
    }
  }
  
  // Linear scale - start from linearMin if provided, otherwise auto
  const range = maxVal - minVal
  const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
  const autoMin = Math.max(0, minVal - padding)
  return {
    min: linearMin !== undefined ? linearMin : autoMin,
    max: maxVal + padding,
  }
}

const computeBinnedThroughput = (peakTimes: number[], binWidthSeconds = 10) => {
  if (!peakTimes.length) {
    return { binCentersMinutes: [] as number[], throughputPerBin: [] as number[] }
  }

  const timesSorted = [...peakTimes].sort((a, b) => a - b)
  const start = timesSorted[0]
  const end = timesSorted[timesSorted.length - 1]
  const binCount = Math.max(1, Math.ceil((end - start) / binWidthSeconds))

  const counts = new Array(binCount).fill(0)
  for (const t of timesSorted) {
    const idx = Math.min(binCount - 1, Math.floor((t - start) / binWidthSeconds))
    counts[idx] += 1
  }

  const binCentersMinutes: number[] = []
  const throughputPerBin: number[] = []
  for (let i = 0; i < binCount; i += 1) {
    const binStart = start + i * binWidthSeconds
    const binCenterSec = binStart + binWidthSeconds / 2
    binCentersMinutes.push(Math.max(0, binCenterSec / 60))
    // Normalize to peaks per second by dividing by bin width
    // This ensures consistent units regardless of bin width
    throughputPerBin.push(counts[i] / binWidthSeconds)
  }

  return { binCentersMinutes, throughputPerBin }
}

const computeMovingAverage = (values: number[], windowSize = 5): number[] => {
  if (!values.length || windowSize <= 1) return values.slice()
  const result: number[] = []
  for (let i = 0; i < values.length; i += 1) {
    let sum = 0
    let count = 0
    for (let j = Math.max(0, i - windowSize + 1); j <= i; j += 1) {
      sum += values[j]
      count += 1
    }
    result.push(count > 0 ? sum / count : 0)
  }
  return result
}

export function AnalyzeTimeSeries({
  peakTimes = [],
  peakAmplitudes = [],
  peakIntervalsMs = [],
  peakWidths = [],
  timeResolution,
  className = "",
  binWidthSeconds = 10,
  rollingMeanWindow = 10,
}: AnalyzeTimeSeriesProps) {
  const { theme } = useTheme()
  const isDark = theme === "dark"

  const [sharedXRange, setSharedXRange] = useState<Range>(null)
  const [ampScale, setAmpScale] = useState<ScaleType>("log")
  const [widthScale, setWidthScale] = useState<ScaleType>("log")
  
  // Use refs to track previous scale to avoid unnecessary recalculations
  const prevAmpScaleRef = useRef<ScaleType>(ampScale)
  const prevWidthScaleRef = useRef<ScaleType>(widthScale)

  // Prepare all data points (no downsampling)
  const xTimesMinutes = useMemo(
    () => peakTimes.map((t) => Math.max(0, t / 60)),
    [peakTimes],
  )

  const widthsMs = useMemo(() => {
    if (!peakWidths.length) return [] as number[]
    const scale =
      typeof timeResolution === "number" && Number.isFinite(timeResolution)
        ? timeResolution * 1000
        : 1
    return peakWidths.map((w) => w * scale)
  }, [peakWidths, timeResolution])


  // Compute dynamic opacity based on number of points
  const pointOpacity = useMemo(
    () => computePointOpacity(peakTimes.length, isDark),
    [peakTimes.length, isDark]
  )
  
  // Cooler scatter dots with dynamic opacity; distinct mean line for contrast
  const pointColor = isDark 
    ? `rgba(203, 213, 225, ${pointOpacity})` 
    : `rgba(15, 23, 42, ${pointOpacity})` // slate-300 tint
  const rollingMeanColor = isDark ? "#fb923c" : "#f97316" // accent orange for rolling mean

  // Compute rolling mean for prominence data
  const amplitudeRollingMean = useMemo(() => {
    if (!peakAmplitudes.length) return [] as number[]
    return computeMovingAverage(peakAmplitudes as number[], rollingMeanWindow)
  }, [peakAmplitudes, rollingMeanWindow])

  // Compute rolling mean for width data
  const widthRollingMean = useMemo(() => {
    if (!widthsMs.length) return [] as number[]
    return computeMovingAverage(widthsMs, rollingMeanWindow)
  }, [widthsMs, rollingMeanWindow])

  const amplitudeSeries: Series[] = useMemo(
    () => [
      {
        label: "Peak prominence",
        color: pointColor,
        width: 0,
        data: peakAmplitudes as number[],
        points: true,
        pointSize: 3,
      },
      {
        label: `Rolling mean prominence`,
        color: rollingMeanColor,
        width: 2,
        data: amplitudeRollingMean,
      },
    ],
    [pointColor, peakAmplitudes, amplitudeRollingMean, rollingMeanColor],
  )

  const widthSeries: Series[] = useMemo(
    () => [
      {
        label: "Peak width (ms)",
        color: pointColor,
        width: 0,
        data: widthsMs,
        points: true,
        pointSize: 3,
      },
      {
        label: `Rolling mean width`,
        color: rollingMeanColor,
        width: 2,
        data: widthRollingMean,
      },
    ],
    [pointColor, widthsMs, widthRollingMean, rollingMeanColor],
  )

  // Compute ranges - linear starts at 0, log auto-detects from data
  // Avoid array concatenation by computing range from both arrays separately
  const amplitudeRange = useMemo(() => {
    // Compute min/max from both arrays without concatenation
    let minVal = Infinity
    let maxVal = -Infinity
    
    const processArray = (arr: number[]) => {
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i]
        if (Number.isFinite(v)) {
          if (v < minVal) minVal = v
          if (v > maxVal) maxVal = v
        }
      }
    }
    
    processArray(peakAmplitudes as number[])
    processArray(amplitudeRollingMean)
    
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null
    
    // Create a small array with just min/max for computeRange
    return computeRange([minVal, maxVal], ampScale, ampScale === "linear" ? 0 : undefined)
  }, [peakAmplitudes, amplitudeRollingMean, ampScale])
  
  const widthRange = useMemo(() => {
    let minVal = Infinity
    let maxVal = -Infinity
    
    const processArray = (arr: number[]) => {
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i]
        if (Number.isFinite(v)) {
          if (v < minVal) minVal = v
          if (v > maxVal) maxVal = v
        }
      }
    }
    
    processArray(widthsMs)
    processArray(widthRollingMean)
    
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null
    
    return computeRange([minVal, maxVal], widthScale, widthScale === "linear" ? 0 : undefined)
  }, [widthsMs, widthRollingMean, widthScale])
  
  const { binCentersMinutes, throughputPerBin } = useMemo(
    () => computeBinnedThroughput(peakTimes, binWidthSeconds),
    [peakTimes, binWidthSeconds],
  )
  const movingAverage = useMemo(
    () => computeMovingAverage(throughputPerBin, 5),
    [throughputPerBin],
  )

  const throughputBarSeries: Series[] = useMemo(
    () => [
      {
        label: `Rolling mean throughput (5-bin)`,
        color: isDark ? "#fb923c" : "#f97316",  // accent orange
        width: 2,
        data: movingAverage,
      },
      {
        label: "Peak throughput",
        color: isDark ? "rgba(148, 163, 184, 0.6)" : "rgba(100, 116, 139, 0.65)",  // slate/gray
        width: 0,
        data: throughputPerBin,
        bar: true,
      },
    ],
    [isDark, throughputPerBin, movingAverage],
  )

  const throughputBarRange = useMemo(
    () => computeRange(throughputPerBin.concat(movingAverage), "linear", 0),
    [throughputPerBin, movingAverage],
  )

  const handleSharedRangeChange = useCallback(
    (range: { min: number; max: number }) => {
      setSharedXRange((prev) => {
        if (!prev) return range
        // Slightly larger threshold (1e-3) to reduce unnecessary updates during rapid zooming
        if (Math.abs(prev.min - range.min) < 1e-3 && Math.abs(prev.max - range.max) < 1e-3) {
          return prev
        }
        return range
      })
    },
    [],
  )

  const handleResetZoom = useCallback(() => {
    setSharedXRange(null)
  }, [])

  if (!peakTimes.length || !peakAmplitudes.length) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <p className="text-sm text-muted-foreground">No detected peaks to analyze</p>
      </div>
    )
  }

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {/* Prominence */}
      <div className="border rounded-lg bg-card/60 shadow-sm overflow-hidden">
        <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/20">
          <span className="text-xs font-medium text-muted-foreground">Prominence</span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              className={axisButtonClass(ampScale === "linear")}
              onClick={() => setAmpScale("linear")}
            >
              Linear
            </button>
            <button
              type="button"
              className={axisButtonClass(ampScale === "log")}
              onClick={() => setAmpScale("log")}
            >
              Log
            </button>
          </div>
        </div>
        <div className="p-1">
          <UPlotChart
            xData={xTimesMinutes}
            series={amplitudeSeries}
            xLabel="Time (min)"
            yLabel="Prominence"
            height={200}
            xRange={sharedXRange || undefined}
            onXRangeChange={handleSharedRangeChange}
            onResetZoom={handleResetZoom}
            yScaleType={ampScale}
            yRange={amplitudeRange || undefined}
          />
        </div>
      </div>

      {/* Width */}
      <div className="border rounded-lg bg-card/60 shadow-sm overflow-hidden">
        <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/20">
          <span className="text-xs font-medium text-muted-foreground">Width (ms)</span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              className={axisButtonClass(widthScale === "linear")}
              onClick={() => setWidthScale("linear")}
            >
              Linear
            </button>
            <button
              type="button"
              className={axisButtonClass(widthScale === "log")}
              onClick={() => setWidthScale("log")}
            >
              Log
            </button>
          </div>
        </div>
        <div className="p-1">
          <UPlotChart
            xData={xTimesMinutes}
            series={widthSeries}
            xLabel="Time (min)"
            yLabel="Width (ms)"
            height={200}
            xRange={sharedXRange || undefined}
            onXRangeChange={handleSharedRangeChange}
            onResetZoom={handleResetZoom}
            yScaleType={widthScale}
            yRange={widthRange || undefined}
          />
        </div>
      </div>

      {/* Binned throughput bar + moving average */}
      {binCentersMinutes.length > 0 && throughputPerBin.length > 0 && (
        <div className="border rounded-lg bg-card/60 shadow-sm overflow-hidden">
          <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/20">
            <span className="text-xs font-medium text-muted-foreground">Peak Throughput</span>
            <span className="text-[10px] text-muted-foreground/70">
              Scroll to zoom, double-click to reset
            </span>
          </div>
          <div className="p-1">
            <UPlotChart
              xData={binCentersMinutes}
              series={throughputBarSeries}
              xLabel="Time (min)"
              yLabel="Peaks/sec"
              height={200}
              xRange={sharedXRange || undefined}
              onXRangeChange={handleSharedRangeChange}
              onResetZoom={handleResetZoom}
              yScaleType="linear"
              yRange={throughputBarRange || undefined}
              enableYAxisZoom={true}
            />
          </div>
        </div>
      )}
    </div>
  )
}
