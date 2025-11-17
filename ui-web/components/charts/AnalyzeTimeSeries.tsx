"use client"

import { useMemo, useState, useCallback } from "react"
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

type ScaleType = "linear" | "log"

type Range = { min: number; max: number } | null

const axisButtonClass = (active: boolean) =>
  `px-2.5 py-1 rounded-md border text-[10px] font-medium transition-colors ${
    active
      ? "bg-primary border-primary text-primary-foreground shadow-sm"
      : "border-border/50 text-muted-foreground hover:text-foreground hover:border-border hover:bg-accent/50"
  }`

const computeRange = (values: number[], scale: ScaleType): Range => {
  const finite = values.filter((v) => Number.isFinite(v))
  if (!finite.length) return null
  let minVal = Math.min(...finite)
  let maxVal = Math.max(...finite)
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
    return null
  }
  if (scale === "log") {
    const positive = finite.filter((v) => v > 0)
    if (!positive.length) return null
    minVal = Math.min(...positive)
    maxVal = Math.max(...positive)
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null
    const paddingFactor = 0.1
    const minPadded = minVal / (1 + paddingFactor)
    const maxPadded = maxVal * (1 + paddingFactor)
    return {
      min: Math.max(minPadded, minVal * 0.5),
      max: maxPadded,
    }
  }
  const range = maxVal - minVal
  const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
  return {
    min: Math.max(0, minVal - padding),
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

  const instantaneousThroughput = useMemo(() => {
    if (!peakIntervalsMs.length) return [] as number[]
    return peakIntervalsMs.map((dt) =>
      dt > 0 && Number.isFinite(dt) ? 1000 / dt : NaN,
    )
  }, [peakIntervalsMs])

  // Cooler scatter dots with lower opacity; distinct mean line for contrast
  const pointColor = isDark ? "rgba(203, 213, 225, 0.1)" : "rgba(15,23,42,0.5)" // slate-300 tint
  const rollingMeanColor = isDark ? "#60a5fa" : "#3b82f6" // blue color used throughout the app

  // Compute rolling mean for amplitude data
  const amplitudeRollingMean = useMemo(() => {
    if (!peakAmplitudes.length) return [] as number[]
    return computeMovingAverage(peakAmplitudes, rollingMeanWindow)
  }, [peakAmplitudes, rollingMeanWindow])

  // Compute rolling mean for width data
  const widthRollingMean = useMemo(() => {
    if (!widthsMs.length) return [] as number[]
    return computeMovingAverage(widthsMs, rollingMeanWindow)
  }, [widthsMs, rollingMeanWindow])

  const amplitudeSeries: Series[] = useMemo(
    () => [
      {
        label: "Peak Amplitude",
        color: pointColor,
        width: 0,
        data: peakAmplitudes,
        points: true,
        pointSize: 3,
      },
      {
        label: `Rolling Mean (${rollingMeanWindow}-point)`,
        color: rollingMeanColor,
        width: 2,
        data: amplitudeRollingMean,
      },
    ],
    [pointColor, peakAmplitudes, amplitudeRollingMean, rollingMeanColor, rollingMeanWindow],
  )

  const widthSeries: Series[] = useMemo(
    () => [
      {
        label: "Peak Width (ms)",
        color: pointColor,
        width: 0,
        data: widthsMs,
        points: true,
        pointSize: 3,
      },
      {
        label: `Rolling Mean (${rollingMeanWindow}-point)`,
        color: rollingMeanColor,
        width: 2,
        data: widthRollingMean,
      },
    ],
    [pointColor, widthsMs, widthRollingMean, rollingMeanColor, rollingMeanWindow],
  )

  const amplitudeRange = useMemo(
    () => computeRange(peakAmplitudes.concat(amplitudeRollingMean), ampScale),
    [peakAmplitudes, amplitudeRollingMean, ampScale],
  )
  const widthRange = useMemo(
    () => computeRange(widthsMs.concat(widthRollingMean), widthScale),
    [widthsMs, widthRollingMean, widthScale],
  )
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
        label: "5-point Moving Average",
        color: isDark ? "#f97316" : "#ea580c",
        width: 2,
        data: movingAverage,
      },
      {
        label: "Peak Throughput",
        color: isDark ? "rgba(96,165,250,0.7)" : "rgba(59,130,246,0.7)",
        width: 0,
        data: throughputPerBin,
        bar: true,
      },
    ],
    [isDark, throughputPerBin, movingAverage],
  )

  const throughputBarRange = useMemo(
    () => computeRange(throughputPerBin.concat(movingAverage), "linear"),
    [throughputPerBin, movingAverage],
  )

  const handleSharedRangeChange = useCallback(
    (range: { min: number; max: number }) => {
      setSharedXRange((prev) => {
        if (!prev) return range
        if (Math.abs(prev.min - range.min) < 1e-4 && Math.abs(prev.max - range.max) < 1e-4) {
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
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Amplitude */}
      <div className="border rounded-lg bg-card/60 shadow-sm p-3">
        <div className="flex items-center justify-end mb-2">
          <div className="flex items-center gap-2 text-[10px]">
            <span className="text-muted-foreground">Y Scale:</span>
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
              Log10
            </button>
          </div>
        </div>
        <UPlotChart
          xData={xTimesMinutes}
          series={amplitudeSeries}
          xLabel="Time (min)"
          yLabel="Amplitude"
          height={200}
          xRange={sharedXRange || undefined}
          onXRangeChange={handleSharedRangeChange}
          onResetZoom={handleResetZoom}
          yScaleType={ampScale}
          yRange={amplitudeRange || undefined}
        />
      </div>

      {/* Width */}
      <div className="border rounded-lg bg-card/60 shadow-sm p-3">
        <div className="flex items-center justify-end mb-2">
          <div className="flex items-center gap-2 text-[10px]">
            <span className="text-muted-foreground">Y Scale:</span>
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
              Log10
            </button>
          </div>
        </div>
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

      {/* Binned throughput bar + moving average */}
      {binCentersMinutes.length > 0 && throughputPerBin.length > 0 && (
        <div className="border rounded-lg bg-card/60 shadow-sm p-3">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-medium">Peak Throughput</p>
            <p className="text-[10px] text-muted-foreground">
              Scroll to zoom, double-click to reset
            </p>
          </div>
          <UPlotChart
            xData={binCentersMinutes}
            series={throughputBarSeries}
            xLabel="Time (min)"
            yLabel="Peaks per second"
            height={220}
            xRange={sharedXRange || undefined}
            onXRangeChange={handleSharedRangeChange}
            onResetZoom={handleResetZoom}
            yScaleType="linear"
            yRange={throughputBarRange || undefined}
            enableYAxisZoom={true}
          />
        </div>
      )}
    </div>
  )
}
