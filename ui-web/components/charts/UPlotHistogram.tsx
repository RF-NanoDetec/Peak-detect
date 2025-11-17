"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "@/hooks/use-theme"
import uPlot from "uplot"

const UplotReact = dynamic(() => import("react-uplot").then((mod) => mod.UPlot), { ssr: false })

interface HistogramData {
  bins: number[]
  counts: number[]
  bin_edges?: number[]
}

type AxisScaleType = "linear" | "log"
type YAxisScaleType = "linear" | "log"

interface UPlotHistogramProps {
  data: HistogramData
  xLabel: string
  yLabel?: string
  color?: string
  xScaleType?: AxisScaleType
  yScaleType?: YAxisScaleType
  height?: number
  className?: string
}

const formatAxisNumber = (value: number | null | undefined): string => {
  if (value == null || Number.isNaN(value)) return ""
  if (value === 0) return "0"
  const abs = Math.abs(value)
  if (abs < 0.001 || abs > 10000) return value.toExponential(1)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  if (abs >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

const SCALE_DISTRIBUTIONS = {
  LOG: 3,
  ARCSINH: 4,
} as const

const defaultRange = (scaleType: AxisScaleType) =>
  scaleType === "linear" ? { min: 0, max: 100 } : { min: 0.1, max: 1000 }

export function UPlotHistogram({
  data,
  xLabel,
  yLabel = "Frequency",
  color,
  xScaleType = "linear",
  yScaleType = "linear",
  height = 200,
  className = "",
}: UPlotHistogramProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const uPlotInstanceRef = useRef<uPlot | null>(null)
  const [width, setWidth] = useState<number>(600)
  const [mounted, setMounted] = useState(false)
  const [chartReady, setChartReady] = useState(false)
  const { theme } = useTheme()
  const isDark = theme === "dark"

  const barColor = color || (isDark ? "#5b9bd5" : "#3b82f6")
  const edgeColor = isDark ? "#3a7ba5" : "#2563eb"

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const w = entries[0].contentRect.width
      setWidth(Math.max(200, Math.floor(w)))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const opts: uPlot.Options = useMemo(() => {
    const gridColor = isDark ? "rgba(248,250,252,0.10)" : "rgba(15,23,42,0.06)"
    const axisColor = isDark ? "rgba(248,250,252,0.88)" : "rgba(15,23,42,0.65)"

    const xData = data.bins || []
    const yData = data.counts || []
    const binEdges = data.bin_edges || []

    const xScaleBase: uPlot.Scale = { time: false }
    if (xScaleType === "log") {
      xScaleBase.distr = SCALE_DISTRIBUTIONS.LOG
      xScaleBase.clamp = (_self, val) => Math.max(val, 1e-9)
      xScaleBase.log = 10
    }

    const yScaleBase: uPlot.Scale = {}
    if (yScaleType === "log") {
      yScaleBase.distr = SCALE_DISTRIBUTIONS.LOG
      yScaleBase.clamp = (_self, val) => Math.max(val, 1)
      yScaleBase.log = 10
      yScaleBase.range = (_u: uPlot, dataMin: number, dataMax: number) => {
        let min = Number.isFinite(dataMin) ? dataMin : NaN
        let max = Number.isFinite(dataMax) ? dataMax : NaN
        if (!Number.isFinite(min) || min <= 0) min = 1
        if (!Number.isFinite(max) || max <= min) max = min * 10
        return [min, max]
      }
    } else {
      yScaleBase.auto = false
      yScaleBase.min = 0
    }

    const effectiveEdges = binEdges.length > 0 ? binEdges : xData
    const validEdges = effectiveEdges.filter(v =>
      Number.isFinite(v) && (xScaleType !== "log" || v > 0)
    )

    const edgeRange = () => {
      if (validEdges.length >= 2) {
        const minVal = Math.min(...validEdges)
        const maxVal = Math.max(...validEdges)
        if (xScaleType === "log" && minVal > 0) {
          const padding = 0.1
          return {
            min: minVal / (1 + padding),
            max: maxVal * (1 + padding),
          }
        }
        const range = maxVal - minVal
        const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
        return {
          min: Math.max(0, minVal - padding),
          max: maxVal + padding,
        }
      }
      return defaultRange(xScaleType)
    }

    return {
      width,
      height,
      scales: {
        x: {
          ...xScaleBase,
          ...(validEdges.length >= 2 ? edgeRange() : defaultRange(xScaleType)),
        },
        y: {
          ...yScaleBase,
        },
      },
      axes: [
        {
          stroke: axisColor,
          grid: { stroke: gridColor, width: 1 },
          label: xLabel,
          labelSize: 20,
          font: "600 11px 'Inter', 'Segoe UI', system-ui, sans-serif",
          labelFont: "600 12px 'Inter', 'Segoe UI', system-ui, sans-serif",
          size: 50,
          values: (_u: uPlot, vals: number[]) => vals.map(formatAxisNumber),
        },
        {
          stroke: axisColor,
          grid: { stroke: gridColor, width: 1 },
          label: yLabel,
          labelSize: 30,
          font: "600 11px 'Inter', 'Segoe UI', system-ui, sans-serif",
          labelFont: "600 12px 'Inter', 'Segoe UI', system-ui, sans-serif",
          size: 60,
          values: (_u: uPlot, vals: number[]) => vals.map(formatAxisNumber),
        },
      ],
      legend: {
        show: false,
      },
      series: [
        {},
        {
          label: "",
          stroke: barColor,
          width: 0,
        },
      ],
      hooks: {
        ready: [
          (u: uPlot) => {
            uPlotInstanceRef.current = u
            setChartReady(true)
          },
        ],
        draw: [
          (u: uPlot) => {
            const ctx = u.ctx
            const xValues = (u.data[0] as number[]) || []
            const yValues = (u.data[1] as number[]) || []
            const edges = data.bin_edges || []

            ctx.save()
            ctx.lineWidth = 1
            ctx.fillStyle = barColor
            ctx.strokeStyle = edgeColor

            for (let i = 0; i < xValues.length; i += 1) {
              const count = yValues[i]
              if (!Number.isFinite(count)) continue

              const center =
                edges.length > 0
                  ? (xScaleType === "linear"
                      ? (edges[i] + edges[i + 1]) / 2
                      : Math.sqrt(edges[i] * edges[i + 1]))
                  : xValues[i]

              let barWidthPx = 0
              if (edges.length > 0) {
                const left = u.valToPos(edges[i], "x", true)
                const right = u.valToPos(edges[i + 1], "x", true)
                barWidthPx = Math.abs(right - left)
              } else if (xValues.length > 1) {
                const prev = xValues[i - 1] ?? xValues[i]
                const next = xValues[i + 1] ?? xValues[i]
                const leftPx = u.valToPos(prev, "x", true)
                const rightPx = u.valToPos(next, "x", true)
                barWidthPx = Math.abs(rightPx - leftPx) * 0.5
              } else {
                barWidthPx = Math.max(2, width * 0.05)
              }

              if (barWidthPx <= 0) continue

              const scaleMin = u.scales.y?.min ?? 0
              // For log scale, clamp to minimum of 1 to prevent negative bars
              const countValue = yScaleType === "log" ? Math.max(count, 1) : count
              const baseValue = yScaleType === "log" ? Math.max(scaleMin, 1) : scaleMin

              const xPx = u.valToPos(center, "x", true)
              const yPx = u.valToPos(countValue, "y", true)
              const zeroPx = u.valToPos(baseValue, "y", true)

              const barHeight = Math.abs(zeroPx - yPx)
              const barLeft = xPx - barWidthPx / 2
              const barTop = Math.min(zeroPx, yPx)

              if (barHeight > 0) {
                ctx.fillRect(barLeft, barTop, barWidthPx, barHeight)
                ctx.strokeRect(barLeft, barTop, barWidthPx, barHeight)
              }
            }

            ctx.restore()
          },
        ],
      },
    }
  }, [width, height, isDark, data, xLabel, yLabel, barColor, edgeColor, xScaleType, yScaleType])

  const chartData = useMemo(() => {
    const xValues = (data.bins || []).filter(v => Number.isFinite(v))
    const yValues = (data.counts || []).filter(v => Number.isFinite(v))
    
    // Ensure arrays have the same length
    const minLength = Math.min(xValues.length, yValues.length)
    return [
      xValues.slice(0, minLength), 
      yValues.slice(0, minLength)
    ] as uPlot.AlignedData
  }, [data])

  useEffect(() => {
    if (!chartReady || !uPlotInstanceRef.current || !data.bins || data.bins.length === 0) {
      return
    }

    const u = uPlotInstanceRef.current
    const xVals = data.bins || []
    const edges = data.bin_edges || []

    const calcLinearPadding = (minVal: number, maxVal: number) => {
      const range = maxVal - minVal
      const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
      return { min: Math.max(0, minVal - padding), max: maxVal + padding }
    }

    const applyXScale = (minVal: number, maxVal: number) => {
      if (xScaleType === "linear") {
        u.setScale("x", calcLinearPadding(minVal, maxVal))
      } else if (xScaleType === "log" && minVal > 0) {
        const padding = 0.1
        u.setScale("x", {
          min: minVal / (1 + padding),
          max: maxVal * (1 + padding),
        })
      }
    }

    if (edges.length >= 2) {
      const validEdges = edges.filter(v => Number.isFinite(v) && (xScaleType !== "log" || v > 0))
      if (validEdges.length >= 2) {
        applyXScale(Math.min(...validEdges), Math.max(...validEdges))
      }
    } else if (xVals.length >= 2) {
      const validX = xVals.filter(v => Number.isFinite(v) && (xScaleType !== "log" || v > 0))
      if (validX.length >= 2) {
        applyXScale(Math.min(...validX), Math.max(...validX))
      }
    }

    const yVals = data.counts || []
    if (yVals.length > 0) {
      const maxCount = Math.max(...yVals)
      const positiveCounts = yVals.filter(v => Number.isFinite(v) && v > 0)
      if (yScaleType === "log") {
        // For log scale, ensure minimum is at least 1 to avoid negative bars
        const minPositive = positiveCounts.length > 0 ? Math.min(...positiveCounts) : 1
        const effectiveMin = Math.max(1, minPositive * 0.5)
        const effectiveMax = Math.max(10, maxCount * 1.5)
        u.setScale("y", {
          min: effectiveMin,
          max: effectiveMax,
        })
      } else {
        u.setScale("y", { min: 0, max: Math.max(1, maxCount * 1.1) })
      }
    }
  }, [chartReady, data, xScaleType, yScaleType])

  const hasData = data.bins && data.bins.length > 0 && data.counts && data.counts.length > 0

  if (!hasData) {
    return (
      <div ref={containerRef} className={`flex items-center justify-center ${className}`} style={{ height }}>
        <p className="text-sm text-muted-foreground">No data available</p>
      </div>
    )
  }

  if (!mounted) {
    return (
      <div ref={containerRef} className={className} style={{ height, minHeight: 0 }} />
    )
  }

  return (
    <div ref={containerRef} className={className} style={{ minHeight: 0 }}>
      <UplotReact key={`${xScaleType}-${yScaleType}`} options={opts} data={chartData} />
    </div>
  )
}
