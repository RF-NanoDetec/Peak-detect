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
  verticalLines?: Array<{ value: number; color?: string; label?: string }>
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
  verticalLines = [],
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

    // Filter valid edges based on scale type
    const effectiveEdges = binEdges.length > 0 ? binEdges : xData
    const validEdges = effectiveEdges.filter(v =>
      Number.isFinite(v) && (xScaleType !== "log" || v > 0)
    )

    // Filter valid counts based on scale type
    const validCounts = yData.filter(v =>
      Number.isFinite(v) && (yScaleType !== "log" || v > 0)
    )

    // X Scale Configuration
    const xScaleBase: uPlot.Scale = { time: false }
    if (xScaleType === "log") {
      xScaleBase.distr = SCALE_DISTRIBUTIONS.LOG
      xScaleBase.clamp = (_self, val) => Math.max(val, 1e-9)
      xScaleBase.log = 10
      // Use range function for log scale instead of explicit min/max
      xScaleBase.range = (_u: uPlot, dataMin: number, dataMax: number) => {
        // Get valid edges for range calculation
        if (validEdges.length >= 2) {
          const minVal = Math.min(...validEdges)
          const maxVal = Math.max(...validEdges)
          if (minVal > 0 && maxVal > minVal) {
            const padding = 0.1
            return [minVal / (1 + padding), maxVal * (1 + padding)]
          }
        }
        // Fallback to safe defaults for log scale
        return [0.1, 1000]
      }
    } else {
      // For linear scale, we can set explicit min/max
      if (validEdges.length >= 2) {
        const minVal = Math.min(...validEdges)
        const maxVal = Math.max(...validEdges)
        const range = maxVal - minVal
        const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
        xScaleBase.min = Math.max(0, minVal - padding)
        xScaleBase.max = maxVal + padding
      } else {
        xScaleBase.min = 0
        xScaleBase.max = 100
      }
    }

    // Y Scale Configuration
    const yScaleBase: uPlot.Scale = {}
    if (yScaleType === "log") {
      yScaleBase.distr = SCALE_DISTRIBUTIONS.LOG
      yScaleBase.clamp = (_self, val) => Math.max(val, 1)
      yScaleBase.log = 10
      yScaleBase.range = (_u: uPlot, dataMin: number, dataMax: number) => {
        // Use valid counts for range calculation
        if (validCounts.length > 0) {
          const minPositive = Math.min(...validCounts)
          const maxPositive = Math.max(...validCounts)
          if (minPositive > 0 && maxPositive > minPositive) {
            const effectiveMin = Math.max(1, minPositive * 0.5)
            const effectiveMax = Math.max(10, maxPositive * 1.5)
            return [effectiveMin, effectiveMax]
          }
        }
        // Fallback to safe defaults for log scale
        return [1, 100]
      }
    } else {
      // For linear Y scale, start from 0
      yScaleBase.auto = false
      yScaleBase.min = 0
      if (validCounts.length > 0) {
        const maxCount = Math.max(...validCounts)
        yScaleBase.max = Math.max(1, maxCount * 1.1)
      }
    }

    return {
      width,
      height,
      scales: {
        x: xScaleBase,
        y: yScaleBase,
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

            // Draw vertical lines for thresholds
            if (verticalLines.length > 0) {
              ctx.save()
              verticalLines.forEach((line) => {
                const { value, color: lineColor = "#ef4444", label } = line
                
                // Skip if value is not in valid range for scale type
                if (!Number.isFinite(value)) return
                if (xScaleType === "log" && value <= 0) return
                
                const xPx = u.valToPos(value, "x", true)
                const yMin = u.valToPos(u.scales.y?.min ?? 0, "y", true)
                const yMax = u.valToPos(u.scales.y?.max ?? 1, "y", true)
                
                // Draw vertical line
                ctx.strokeStyle = lineColor
                ctx.lineWidth = 2
                ctx.setLineDash([5, 3])
                ctx.beginPath()
                ctx.moveTo(xPx, yMax)
                ctx.lineTo(xPx, yMin)
                ctx.stroke()
                ctx.setLineDash([])
                
                // Draw label if provided
                if (label) {
                  ctx.fillStyle = lineColor
                  ctx.font = "600 10px 'Inter', 'Segoe UI', system-ui, sans-serif"
                  ctx.textAlign = "center"
                  ctx.fillText(label, xPx, yMax - 5)
                }
              })
              ctx.restore()
            }
          },
        ],
      },
    }
  }, [width, height, isDark, data, xLabel, yLabel, barColor, edgeColor, xScaleType, yScaleType, verticalLines])

  const chartData = useMemo(() => {
    const rawX = data.bins || []
    const rawY = data.counts || []
    
    // Filter data based on scale type requirements
    const filteredData: Array<[number, number]> = []
    const minLength = Math.min(rawX.length, rawY.length)
    
    for (let i = 0; i < minLength; i++) {
      const x = rawX[i]
      const y = rawY[i]
      
      // Skip invalid values
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue
      
      // For log scales, skip non-positive values
      if (xScaleType === "log" && x <= 0) continue
      if (yScaleType === "log" && y <= 0) continue
      
      filteredData.push([x, y])
    }
    
    // Separate back into x and y arrays
    const xValues = filteredData.map(([x]) => x)
    const yValues = filteredData.map(([, y]) => y)
    
    return [xValues, yValues] as uPlot.AlignedData
  }, [data, xScaleType, yScaleType])

  // Note: Manual scale updates removed - the component remounts on scale changes
  // via the key prop, so scale configuration in opts is sufficient

  const hasData = data.bins && data.bins.length > 0 && data.counts && data.counts.length > 0

  if (!hasData) {
    return (
      <div ref={containerRef} className={`flex items-center justify-center ${className}`} style={{ height }}>
        <p className="text-sm text-muted-foreground">No data available</p>
      </div>
    )
  }

  // Check if filtered data is empty (e.g., all values were negative/zero for log scale)
  const hasValidData = chartData[0].length > 0 && chartData[1].length > 0

  if (!hasValidData) {
    return (
      <div ref={containerRef} className={`flex items-center justify-center ${className}`} style={{ height }}>
        <p className="text-sm text-muted-foreground">
          No valid data for {xScaleType === "log" ? "log X" : "linear X"} / {yScaleType === "log" ? "log Y" : "linear Y"} scale
        </p>
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


