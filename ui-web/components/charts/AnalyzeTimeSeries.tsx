"use client"

import { useMemo, useState, useCallback } from "react"
import type { ReactNode } from "react"
import { UPlotChart, type Series } from "./UPlotChart"
import { useTheme } from "@/hooks/use-theme"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

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
  `h-6 px-2 text-[10px] ${active ? "bg-primary text-primary-foreground" : "bg-transparent text-muted-foreground hover:text-foreground"}`

// Safe min/max for large arrays - avoids stack overflow from spread operator
const safeMinMax = (arr: number[]): { min: number; max: number } | null => {
  if (arr.length === 0) return null
  let min = arr[0]
  let max = arr[0]
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i]
    if (arr[i] > max) max = arr[i]
  }
  return { min, max }
}

const computeRange = (values: number[], scale: ScaleType, minLimit?: number): Range => {
  const finite = values.filter((v) => Number.isFinite(v))
  if (!finite.length) return null
  
  const result = safeMinMax(finite)
  if (!result) return null
  let { min: minVal, max: maxVal } = result
  
  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
    return null
  }

  if (scale === "log") {
    const positive = finite.filter((v) => v > 0)
    if (!positive.length) return null
    const posResult = safeMinMax(positive)
    if (!posResult) return null
    minVal = posResult.min
    maxVal = posResult.max
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return null
    
    const paddingFactor = 0.1
    const maxPadded = maxVal * (1 + paddingFactor)
    
    let finalMin = Math.max(minVal / (1 + paddingFactor), minVal * 0.5);
    
    if (minLimit !== undefined) {
        finalMin = minLimit
    }
    
    return {
      min: finalMin,
      max: maxPadded,
    }
  }
  
  // Linear
  if (minLimit !== undefined) {
      // Ensure max is at least slightly above minLimit to avoid flat line if all data is 0
      const effectiveMax = Math.max(maxVal, minLimit + 1e-6)
      const range = effectiveMax - minLimit
      const padding = range * 0.05
      return { min: minLimit, max: effectiveMax + padding }
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

// Helper component for right-aligned controls layout
const ChartRow = ({ 
  children, 
  label, 
  controls 
}: { 
  children: ReactNode
  label: string
  controls?: ReactNode 
}) => (
  <div className="flex border rounded-lg bg-card/60 shadow-sm overflow-hidden h-[240px]">
    <div className="flex-1 min-w-0 p-1">
      {children}
    </div>
    <div className="w-14 flex flex-col items-center border-l bg-muted/5 py-2 gap-2 flex-shrink-0">
      {controls}
      <div className="flex-1 flex items-center justify-center writing-mode-vertical">
        <span className="transform -rotate-90 whitespace-nowrap text-xs font-medium text-muted-foreground tracking-tight" style={{ writingMode: 'vertical-rl' }}>
          {label}
        </span>
      </div>
    </div>
  </div>
)

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

  const amplitudeRange = useMemo(
    () => computeRange(
      (peakAmplitudes as number[]).concat(amplitudeRollingMean), 
      ampScale, 
      ampScale === "log" ? 1 : 0 // Min limit: 1 for log, 0 for linear
    ),
    [peakAmplitudes, amplitudeRollingMean, ampScale],
  )
  
  const widthRange = useMemo(
    () => computeRange(
      widthsMs.concat(widthRollingMean), 
      widthScale,
      widthScale === "log" ? 0.1 : 0 // Min limit: 0.1 for log, 0 for linear
    ),
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
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Prominence */}
      <ChartRow
        label="Prominence"
        controls={
          <div className="flex flex-col gap-1">
             <Button 
               variant="ghost" 
               size="sm" 
               className={cn("h-6 px-1 text-[10px]", ampScale === "linear" && "bg-accent text-accent-foreground")}
               onClick={() => setAmpScale("linear")}
             >
               Lin
             </Button>
             <Button 
               variant="ghost" 
               size="sm" 
               className={cn("h-6 px-1 text-[10px]", ampScale === "log" && "bg-accent text-accent-foreground")}
               onClick={() => setAmpScale("log")}
             >
               Log
             </Button>
          </div>
        }
      >
        <UPlotChart
          xData={xTimesMinutes}
          series={amplitudeSeries}
          xLabel="Time (min)"
          yLabel="" // Label moved to right
          height={230} // Maximizing height within container
          xRange={sharedXRange || undefined}
          onXRangeChange={handleSharedRangeChange}
          onResetZoom={handleResetZoom}
          yScaleType={ampScale}
          yRange={amplitudeRange || undefined}
        />
      </ChartRow>

      {/* Width */}
      <ChartRow
        label="Width (ms)"
        controls={
          <div className="flex flex-col gap-1">
             <Button 
               variant="ghost" 
               size="sm" 
               className={cn("h-6 px-1 text-[10px]", widthScale === "linear" && "bg-accent text-accent-foreground")}
               onClick={() => setWidthScale("linear")}
             >
               Lin
             </Button>
             <Button 
               variant="ghost" 
               size="sm" 
               className={cn("h-6 px-1 text-[10px]", widthScale === "log" && "bg-accent text-accent-foreground")}
               onClick={() => setWidthScale("log")}
             >
               Log
             </Button>
          </div>
        }
      >
        <UPlotChart
          xData={xTimesMinutes}
          series={widthSeries}
          xLabel="Time (min)"
          yLabel="" // Label moved to right
          height={230}
          xRange={sharedXRange || undefined}
          onXRangeChange={handleSharedRangeChange}
          onResetZoom={handleResetZoom}
          yScaleType={widthScale}
          yRange={widthRange || undefined}
        />
      </ChartRow>

      {/* Binned throughput bar + moving average */}
      {binCentersMinutes.length > 0 && throughputPerBin.length > 0 && (
        <ChartRow
          label="Peaks/sec"
        >
          <UPlotChart
            xData={binCentersMinutes}
            series={throughputBarSeries}
            xLabel="Time (min)"
            yLabel="" // Label moved to right
            height={230}
            xRange={sharedXRange || undefined}
            onXRangeChange={handleSharedRangeChange}
            onResetZoom={handleResetZoom}
            yScaleType="linear"
            yRange={throughputBarRange || undefined}
            enableYAxisZoom={true}
          />
        </ChartRow>
      )}
    </div>
  )
}
