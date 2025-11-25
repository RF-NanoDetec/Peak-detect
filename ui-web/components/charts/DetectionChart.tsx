"use client"

import { useEffect, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { UPlotChart, type Series } from "./UPlotChart"
import { useTheme } from "@/hooks/use-theme"
import { apiClient } from "@/lib/apiClient"

// Maximum points to display for responsive rendering
const MAX_DISPLAY_POINTS = 2500

interface DetectionChartProps {
  previewData?: { time: number[], amplitude: number[] } | null
  resultId?: string
  filteredResultId?: string | null
  peakTimes?: number[]
  peakAmplitudes?: number[]
  className?: string
}

// Min-max decimation for line charts - preserves visual fidelity
const decimateLine = (
  xData: number[],
  yData: number[],
  maxPoints: number
): { x: number[]; y: number[]; decimated: boolean } => {
  if (xData.length <= maxPoints) {
    return { x: xData, y: yData, decimated: false }
  }

  const nBins = Math.max(1, Math.floor(maxPoints / 2))
  const binSize = Math.max(1, Math.floor(xData.length / nBins))
  const outX: number[] = []
  const outY: number[] = []

  for (let bin = 0; bin < nBins; bin++) {
    const start = bin * binSize
    const end = bin === nBins - 1 ? xData.length : Math.min(xData.length, start + binSize)
    if (end <= start) continue

    let minIdx = start
    let maxIdx = start
    let minVal = yData[start]
    let maxVal = yData[start]

    for (let i = start + 1; i < end; i++) {
      if (yData[i] < minVal) {
        minVal = yData[i]
        minIdx = i
      }
      if (yData[i] > maxVal) {
        maxVal = yData[i]
        maxIdx = i
      }
    }

    // Add points in time order to preserve line shape
    if (minIdx <= maxIdx) {
      outX.push(xData[minIdx], xData[maxIdx])
      outY.push(yData[minIdx], yData[maxIdx])
    } else {
      outX.push(xData[maxIdx], xData[minIdx])
      outY.push(yData[maxIdx], yData[minIdx])
    }
  }

  // Remove consecutive duplicates
  const finalX: number[] = []
  const finalY: number[] = []
  for (let i = 0; i < outX.length; i++) {
    if (i === 0 || outX[i] !== outX[i - 1] || outY[i] !== outY[i - 1]) {
      finalX.push(outX[i])
      finalY.push(outY[i])
    }
  }

  return { x: finalX, y: finalY, decimated: true }
}

export function DetectionChart({
  previewData,
  resultId,
  filteredResultId,
  peakTimes = [],
  peakAmplitudes = [],
  className = ""
}: DetectionChartProps) {
  const { theme } = useTheme()
  const isDark = theme === "dark"

  const [xDataFull, setXDataFull] = useState<number[] | null>(null)
  const [yDataFull, setYDataFull] = useState<number[] | null>(null)
  const [fullRes, setFullRes] = useState(false)
  const [loadingFull, setLoadingFull] = useState(false)
  const [fullError, setFullError] = useState<string | null>(null)

  useEffect(() => {
    if (!resultId || !fullRes) {
      setXDataFull(null)
      setYDataFull(null)
      setLoadingFull(false)
      setFullError(null)
      return
    }

    let cancelled = false
    const load = async () => {
      setLoadingFull(true)
      setFullError(null)
      try {
        const base = await apiClient.getResult(resultId, { timeoutMs: 300000 })
        if (cancelled) return
        setXDataFull(base.time.map((t: number) => Math.max(0, t / 60)))
        setYDataFull(base.amplitude)
      } catch (e: any) {
        if (cancelled) return
        setFullError(e?.message || "Failed to load full-resolution data")
        setXDataFull(null)
        setYDataFull(null)
      } finally {
        if (!cancelled) {
          setLoadingFull(false)
        }
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [resultId, fullRes])

  const hasAny = !!previewData || (!!xDataFull && !!yDataFull)

  // Choose full-res if available, else preview
  const xDataRaw = xDataFull ?? (previewData ? previewData.time.map(t => Math.max(0, t / 60)) : [])
  const yDataRaw = yDataFull ?? (previewData ? previewData.amplitude : [])

  // Decimate the data for responsive rendering
  const { xData, yData, isDecimated, originalLength } = useMemo(() => {
    if (!xDataRaw.length) {
      return { xData: [] as number[], yData: [] as number[], isDecimated: false, originalLength: 0 }
    }
    const { x, y, decimated } = decimateLine(xDataRaw, yDataRaw, MAX_DISPLAY_POINTS)
    return { xData: x, yData: y, isDecimated: decimated, originalLength: xDataRaw.length }
  }, [xDataRaw, yDataRaw])

  if (!hasAny) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-sm text-muted-foreground">No data available</p>
      </div>
    )
  }

  // Convert peak times to minutes
  const peakX = (peakTimes || []).map(t => Math.max(0, t / 60))
  const peakY = peakAmplitudes || []

  // Build series: signal line + optional peak scatter
  const series = useMemo(() => {
    console.time("Peak Series Generation")
    const result: Series[] = [
      {
        label: isDecimated ? `Signal (${xData.length.toLocaleString()} pts)` : "Signal",
        color: isDark ? "#94a3b8" : "#64748b",  // slate-400 / slate-500
        width: 1.2,
        data: yData,
      }
    ]

    // Add peaks as a separate series with points
    if (peakX.length > 0) {
      // Create a sparse array matching xData length, with peaks at correct indices
      const peakDataSparse = new Array(xData.length).fill(null)

      // Optimization: Use binary search to find indices instead of linear scan
      // This reduces complexity from O(N*M) to O(M*logN) where N=data points, M=peaks
      const findClosestIndex = (arr: number[], target: number) => {
        let left = 0
        let right = arr.length - 1
        if (arr.length === 0) return -1
        if (target <= arr[0]) return 0
        if (target >= arr[right]) return right

        while (left <= right) {
          const mid = (left + right) >> 1
          if (arr[mid] === target) return mid
          if (arr[mid] < target) left = mid + 1
          else right = mid - 1
        }

        // Check neighbors
        if (left >= arr.length) return right
        if (right < 0) return left
        return (Math.abs(arr[left] - target) < Math.abs(arr[right] - target)) ? left : right
      }

      const peakIndices: number[] = []

      peakX.forEach((px, i) => {
        // Find closest index in xData using binary search
        const idx = findClosestIndex(xData, px)

        // Only add if it's actually close enough (within 0.001 min approx 60ms)
        if (idx !== -1 && Math.abs(xData[idx] - px) < 0.001) {
          peakDataSparse[idx] = peakY[i]
          peakIndices.push(idx)
        }
      })

      // Ensure indices are sorted for binary search in UPlotChart
      peakIndices.sort((a, b) => a - b)

      result.push({
        label: "Peaks",
        color: isDark ? "#f87171" : "#ef4444",
        width: 0,
        data: peakDataSparse as any,
        points: true,
        pointSize: 4,
        dataIndices: peakIndices,
      })
    }
    console.timeEnd("Peak Series Generation")

    return result
  }, [xData, yData, peakX, peakY, isDark, isDecimated])

  return (
    <div className={`flex flex-col h-full w-full ${className}`} style={{ minHeight: 0 }}>
      {/* Peak count info */}
      <div className="px-8 pt-6 pb-2 flex-shrink-0 space-y-1">
        <p className="text-[11px] text-muted-foreground">
          {originalLength.toLocaleString()} points
          {isDecimated && (
            <> • displaying {xData.length.toLocaleString()}</>
          )}
          {" • "}{peakX.length} peaks detected
          {" • "}
          <span className="text-muted-foreground/70">Scroll to zoom • Double-click to reset</span>
        </p>
        <label className="text-[11px] text-muted-foreground flex items-center gap-2">
          <input
            type="checkbox"
            className="h-3 w-3"
            checked={fullRes}
            onChange={(e) => setFullRes(e.target.checked)}
            disabled={!resultId}
          />
          Load full-resolution data
          {fullRes && loadingFull && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
        </label>
        {fullError && (
          <p className="text-[11px] text-red-500">
            {fullError}
          </p>
        )}
      </div>

      {/* uPlot chart */}
      <div className="flex-1 px-4 pb-6" style={{ minHeight: 0 }}>
        <UPlotChart
          xData={xData}
          series={series}
          xLabel="Time (min)"
          yLabel="Counts"
          height={400}
        />
      </div>
    </div>
  )
}
