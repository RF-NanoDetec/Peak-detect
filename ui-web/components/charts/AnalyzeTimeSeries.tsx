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

const MAX_ANALYSIS_POINTS = 50000
const MAX_THROUGHPUT_BINS = 20000

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
  const span = Math.max(0, end - start)
  const requestedBinWidth = Number.isFinite(binWidthSeconds) && binWidthSeconds > 0 ? binWidthSeconds : 10
  const rawBinCount = Math.max(1, Math.ceil(span / requestedBinWidth))
  const binCount = Math.min(rawBinCount, MAX_THROUGHPUT_BINS)
  const effectiveBinWidth = rawBinCount > MAX_THROUGHPUT_BINS && span > 0
    ? span / MAX_THROUGHPUT_BINS
    : requestedBinWidth

  const counts = new Array(binCount).fill(0)
  for (const t of timesSorted) {
    const idx = Math.min(binCount - 1, Math.floor((t - start) / effectiveBinWidth))
    counts[idx] += 1
  }

  const binCentersMinutes: number[] = []
  const throughputPerBin: number[] = []
  for (let i = 0; i < binCount; i += 1) {
    const binStart = start + i * effectiveBinWidth
    const binCenterSec = binStart + effectiveBinWidth / 2
    binCentersMinutes.push(Math.max(0, binCenterSec / 60))
    throughputPerBin.push(counts[i] / effectiveBinWidth)
  }

  return { binCentersMinutes, throughputPerBin }
}

const computeMovingAverage = (values: number[], windowSize = 5): number[] => {
  if (!values.length || windowSize <= 1) return values.slice()
  const normalizedWindow = Math.max(1, Math.floor(windowSize))
  const result: number[] = []
  let sum = 0
  let finiteCount = 0
  for (let i = 0; i < values.length; i += 1) {
    const incoming = values[i]
    if (Number.isFinite(incoming)) {
      sum += incoming
      finiteCount += 1
    }

    const outgoingIndex = i - normalizedWindow
    if (outgoingIndex >= 0) {
      const outgoing = values[outgoingIndex]
      if (Number.isFinite(outgoing)) {
        sum -= outgoing
        finiteCount -= 1
      }
    }

    result.push(finiteCount > 0 ? sum / finiteCount : Number.NaN)
  }
  return result
}

const preparePeakSeries = (
  peakTimes: number[],
  peakAmplitudes: number[],
  peakWidths: number[],
  timeResolution?: number,
) => {
  const scale =
    typeof timeResolution === "number" && Number.isFinite(timeResolution)
      ? timeResolution * 1000
      : 1
  const count = Math.min(peakTimes.length, peakAmplitudes.length)
  const times: number[] = []
  const amplitudes: number[] = []
  const widthsMs: number[] = []

  for (let i = 0; i < count; i += 1) {
    const time = peakTimes[i]
    const amplitude = peakAmplitudes[i]
    if (!Number.isFinite(time) || !Number.isFinite(amplitude)) continue

    times.push(time)
    amplitudes.push(amplitude)

    const width = peakWidths[i]
    widthsMs.push(Number.isFinite(width) ? width * scale : Number.NaN)
  }

  const sourceCount = times.length
  if (sourceCount <= MAX_ANALYSIS_POINTS) {
    return { times, amplitudes, widthsMs, allTimes: times, sourceCount, displayedCount: sourceCount }
  }

  const step = Math.ceil(sourceCount / MAX_ANALYSIS_POINTS)
  const sampledTimes: number[] = []
  const sampledAmplitudes: number[] = []
  const sampledWidthsMs: number[] = []

  for (let i = 0; i < sourceCount; i += step) {
    sampledTimes.push(times[i])
    sampledAmplitudes.push(amplitudes[i])
    sampledWidthsMs.push(widthsMs[i])
  }

  const lastIndex = sourceCount - 1
  if (sampledTimes[sampledTimes.length - 1] !== times[lastIndex]) {
    sampledTimes.push(times[lastIndex])
    sampledAmplitudes.push(amplitudes[lastIndex])
    sampledWidthsMs.push(widthsMs[lastIndex])
  }

  return {
    times: sampledTimes,
    amplitudes: sampledAmplitudes,
    widthsMs: sampledWidthsMs,
    allTimes: times,
    sourceCount,
    displayedCount: sampledTimes.length,
  }
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

  const preparedSeries = useMemo(
    () => preparePeakSeries(peakTimes, peakAmplitudes, peakWidths, timeResolution),
    [peakTimes, peakAmplitudes, peakWidths, timeResolution],
  )

  // Prepare aligned chart data; very large results are sampled for rendering only.
  const xTimesMinutes = useMemo(
    () => preparedSeries.times.map((t) => Math.max(0, t / 60)),
    [preparedSeries.times],
  )


  // Compute dynamic opacity based on number of points
  const pointOpacity = useMemo(
    () => computePointOpacity(preparedSeries.displayedCount, isDark),
    [preparedSeries.displayedCount, isDark]
  )
  
  // Cooler scatter dots with dynamic opacity; distinct mean line for contrast
  const pointColor = isDark 
    ? `rgba(203, 213, 225, ${pointOpacity})` 
    : `rgba(15, 23, 42, ${pointOpacity})` // slate-300 tint
  const rollingMeanColor = isDark ? "#fb923c" : "#f97316" // accent orange for rolling mean

  // Compute rolling mean for prominence data
  const amplitudeRollingMean = useMemo(() => {
    if (!preparedSeries.amplitudes.length) return [] as number[]
    return computeMovingAverage(preparedSeries.amplitudes, rollingMeanWindow)
  }, [preparedSeries.amplitudes, rollingMeanWindow])

  // Compute rolling mean for width data
  const widthRollingMean = useMemo(() => {
    const finiteWidths = preparedSeries.widthsMs.filter(Number.isFinite)
    if (!finiteWidths.length) return [] as number[]

    const rollingValues = computeMovingAverage(preparedSeries.widthsMs, rollingMeanWindow)
    return rollingValues.map((value, index) =>
      Number.isFinite(preparedSeries.widthsMs[index]) ? value : Number.NaN,
    )
  }, [preparedSeries.widthsMs, rollingMeanWindow])

  const amplitudeSeries: Series[] = useMemo(
    () => [
      {
        label: "Peak prominence",
        color: pointColor,
        width: 0,
        data: preparedSeries.amplitudes,
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
    [pointColor, preparedSeries.amplitudes, amplitudeRollingMean, rollingMeanColor],
  )

  const widthSeries: Series[] = useMemo(
    () => [
      {
        label: "Peak width (ms)",
        color: pointColor,
        width: 0,
        data: preparedSeries.widthsMs,
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
    [pointColor, preparedSeries.widthsMs, widthRollingMean, rollingMeanColor],
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
    
    processArray(preparedSeries.amplitudes)
    processArray(amplitudeRollingMean)
    
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null
    
    // Create a small array with just min/max for computeRange
    return computeRange([minVal, maxVal], ampScale, ampScale === "linear" ? 0 : undefined)
  }, [preparedSeries.amplitudes, amplitudeRollingMean, ampScale])
  
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
    
    processArray(preparedSeries.widthsMs)
    processArray(widthRollingMean)
    
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null
    
    return computeRange([minVal, maxVal], widthScale, widthScale === "linear" ? 0 : undefined)
  }, [preparedSeries.widthsMs, widthRollingMean, widthScale])
  
  const { binCentersMinutes, throughputPerBin } = useMemo(
    () => computeBinnedThroughput(preparedSeries.allTimes, binWidthSeconds),
    [preparedSeries.allTimes, binWidthSeconds],
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

  if (!preparedSeries.times.length || !preparedSeries.amplitudes.length) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <p className="text-sm text-muted-foreground">No detected peaks to analyze</p>
      </div>
    )
  }

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {preparedSeries.sourceCount > preparedSeries.displayedCount && (
        <p className="text-[11px] text-muted-foreground px-1">
          Displaying {preparedSeries.displayedCount.toLocaleString()} of {preparedSeries.sourceCount.toLocaleString()} peaks for chart performance.
        </p>
      )}
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
