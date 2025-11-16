"use client"

import { useEffect, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { UPlotChart, type Series } from "./UPlotChart"
import { useTheme } from "@/hooks/use-theme"
import { apiClient } from "@/lib/apiClient"

interface DetectionChartProps {
  previewData?: { time: number[], amplitude: number[] } | null
  resultId?: string
  filteredResultId?: string | null
  peakTimes?: number[]
  peakAmplitudes?: number[]
  className?: string
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

  if (!hasAny) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-sm text-muted-foreground">No data available</p>
      </div>
    )
  }

  // Choose full-res if available, else preview
  const xData = xDataFull ?? (previewData ? previewData.time.map(t => Math.max(0, t / 60)) : [])
  const yData = yDataFull ?? (previewData ? previewData.amplitude : [])

  // Convert peak times to minutes
  const peakX = (peakTimes || []).map(t => Math.max(0, t / 60))
  const peakY = peakAmplitudes || []

  // Build series: signal line + optional peak scatter
  const series = useMemo(() => {
    const result: Series[] = [
      {
        label: "Signal",
        color: isDark ? "#60a5fa" : "#3b82f6",
        width: 1.2,
        data: yData,
      }
    ]

    // Add peaks as a separate series with points
    if (peakX.length > 0) {
      // Create a sparse array matching xData length, with peaks at correct indices
      const peakDataSparse = new Array(xData.length).fill(null)
      peakX.forEach((px, i) => {
        // Find closest index in xData
        const idx = xData.findIndex(x => Math.abs(x - px) < 0.001)
        if (idx !== -1) {
          peakDataSparse[idx] = peakY[i]
        }
      })

      result.push({
        label: "Peaks",
        color: isDark ? "#f87171" : "#ef4444",
        width: 0,
        data: peakDataSparse as any,
        points: true,
        pointSize: 4,
      })
    }

    return result
  }, [xData, yData, peakX, peakY, isDark])

  return (
    <div className={`flex flex-col h-full w-full ${className}`} style={{ minHeight: 0 }}>
      {/* Peak count info */}
      <div className="px-8 pt-6 pb-2 flex-shrink-0 space-y-1">
        <p className="text-[11px] text-muted-foreground">
          {peakX.length} peaks detected
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
