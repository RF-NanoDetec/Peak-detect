"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "next-themes"
import uPlot from "uplot"

const UplotReact = dynamic(() => import("react-uplot").then((mod) => mod.UPlot), { ssr: false })

interface HistogramData {
  bins: number[]
  counts: number[]
  bin_edges?: number[]
}

interface UPlotHistogramProps {
  data: HistogramData
  xLabel: string
  yLabel?: string
  color?: string
  xScaleType?: "linear" | "log"
  height?: number
  className?: string
}

const formatAxisNumber = (value: number): string => {
  if (value === 0) return '0'
  const abs = Math.abs(value)
  if (abs < 0.001 || abs > 10000) return value.toExponential(1)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  if (abs >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

export function UPlotHistogram({
  data,
  xLabel,
  yLabel = "Frequency",
  color,
  xScaleType = "linear",
  height = 200,
  className = "",
}: UPlotHistogramProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const uPlotInstanceRef = useRef<uPlot | null>(null)
  const [width, setWidth] = useState<number>(600)
  const [mounted, setMounted] = useState(false)
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === "dark"

  // Default colors based on theme
  const barColor = color || (isDark ? "#5b9bd5" : "#3b82f6")
  const edgeColor = isDark ? "#3a7ba5" : "#2563eb"

  useEffect(() => {
    setMounted(true)
  }, [])

  // Handle responsive width
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
    const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)"
    const axisColor = isDark ? "#e5e7eb" : "#1f2937"

    // Prepare data - we need x (bins) and y (counts) arrays
    const xData = data.bins || []
    const yData = data.counts || []
    const binEdges = data.bin_edges || []

    return {
      width,
      height,
      scales: {
        x: {
          time: false,
          ...(xScaleType === "log" ? { distr: 2 as const } : {}),
          ...(binEdges.length > 0 ? (() => {
            // Use bin edges for proper range, not bin centers
            const validEdges = binEdges.filter(v => Number.isFinite(v) && (xScaleType === "linear" || v > 0))
            if (validEdges.length >= 2) {
              const minVal = Math.min(...validEdges)
              const maxVal = Math.max(...validEdges)
              // For log scale, ensure min > 0 and add small padding
              if (xScaleType === "log" && minVal > 0) {
                // Add small padding for log scale (multiplicative)
                const padding = 0.1 // 10% padding
                return { 
                  min: minVal / (1 + padding), 
                  max: maxVal * (1 + padding) 
                }
              } else if (xScaleType === "linear") {
                // Add small padding for linear scale (additive)
                const range = maxVal - minVal
                const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1 // 5% padding
                return { 
                  min: Math.max(0, minVal - padding), 
                  max: maxVal + padding 
                }
              }
            }
            // Fallback if edges not valid - use bin centers
            if (xData.length > 0) {
              const validXData = xData.filter(v => Number.isFinite(v) && (xScaleType === "linear" || v > 0))
              if (validXData.length > 0) {
                const minVal = Math.min(...validXData)
                const maxVal = Math.max(...validXData)
                if (xScaleType === "linear") {
                  const range = maxVal - minVal
                  const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
                  return { min: Math.max(0, minVal - padding), max: maxVal + padding }
                }
              }
            }
            // Final fallback - return default range
            return xScaleType === "linear" ? { min: 0, max: 100 } : { min: 0.1, max: 1000 }
          })() : (xData.length > 0 ? (() => {
            // Fallback to bin centers if edges not available
            const validXData = xData.filter(v => Number.isFinite(v) && (xScaleType === "linear" || v > 0))
            if (validXData.length > 0) {
              const minVal = Math.min(...validXData)
              const maxVal = Math.max(...validXData)
              if (xScaleType === "log" && minVal > 0) {
                const padding = 0.1
                return { min: minVal / (1 + padding), max: maxVal * (1 + padding) }
              } else if (xScaleType === "linear") {
                const range = maxVal - minVal
                const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
                return { 
                  min: Math.max(0, minVal - padding), 
                  max: maxVal + padding 
                }
              }
            }
            // If no valid data, return default range to prevent empty display
            return xScaleType === "linear" ? { min: 0, max: 100 } : { min: 0.1, max: 1000 }
          })() : (xScaleType === "linear" ? { min: 0, max: 100 } : { min: 0.1, max: 1000 }))),
        },
        y: {
          auto: false, // Disable auto to ensure proper scaling
          min: 0, // Histograms start at 0
          ...(yData.length > 0 ? {
            max: Math.max(...yData) * 1.1 // Add 10% padding on top
          } : {}),
        },
      },
      axes: [
        {
          stroke: axisColor,
          grid: { stroke: gridColor, width: 1 },
          label: xLabel,
          labelSize: 20,
          labelFont: "12px sans-serif",
          size: 50,
          values: (u: uPlot, vals: number[]) => vals.map(formatAxisNumber),
        },
        {
          stroke: axisColor,
          grid: { stroke: gridColor, width: 1 },
          label: yLabel,
          labelSize: 30,
          labelFont: "12px sans-serif",
          size: 60,
          values: (u: uPlot, vals: number[]) => vals.map(formatAxisNumber),
        },
      ],
      legend: {
        show: false,
      },
      series: [
        {}, // x axis
        {
          label: "",
          stroke: barColor,
          width: 0, // No line, we'll draw bars
        },
      ],
      hooks: {
        ready: [
          (u: uPlot) => {
            uPlotInstanceRef.current = u
          }
        ],
        draw: [
          (u: uPlot) => {
            const ctx = u.ctx
            // Access data from the chart data array (uPlot format: [xData, yData])
            const xData = (u.data[0] as number[]) || []
            const yData = (u.data[1] as number[]) || []
            // Access bin_edges from closure (captured in useMemo dependencies)
            const binEdges = data.bin_edges || []

            if (!xData || !yData || xData.length === 0 || yData.length === 0) {
              return
            }

            ctx.save()
            ctx.fillStyle = barColor
            ctx.strokeStyle = edgeColor
            ctx.lineWidth = 0.5

            // Draw bars
            for (let i = 0; i < xData.length; i++) {
              const binCenter = xData[i]
              const count = yData[i]

              if (!Number.isFinite(binCenter) || !Number.isFinite(count) || count <= 0) {
                continue
              }

              // Calculate bar width from bin edges if available
              // For log scales, this is critical as bin widths vary
              let barWidth: number = 0
              if (binEdges && binEdges.length > i + 1) {
                const leftEdge = binEdges[i]
                const rightEdge = binEdges[i + 1]
                if (Number.isFinite(leftEdge) && Number.isFinite(rightEdge) && leftEdge > 0 && rightEdge > 0) {
                  const leftPx = u.valToPos(leftEdge, "x", true)
                  const rightPx = u.valToPos(rightEdge, "x", true)
                  barWidth = Math.abs(rightPx - leftPx)
                  // Ensure minimum width for visibility
                  if (barWidth < 1) {
                    barWidth = 1
                  }
                }
              }

              // If we couldn't get width from bin edges, estimate from adjacent bins
              // This is less accurate, especially for log scales, but better than nothing
              if (barWidth <= 0) {
                if (i < xData.length - 1) {
                  const nextCenter = xData[i + 1]
                  if (Number.isFinite(nextCenter) && nextCenter > 0 && binCenter > 0) {
                    // For log scale, estimate based on the ratio
                    if (xScaleType === "log") {
                      // Estimate bin width as half the distance to next bin center on each side
                      const leftPx = u.valToPos(binCenter, "x", true)
                      const rightPx = u.valToPos(nextCenter, "x", true)
                      barWidth = Math.abs(rightPx - leftPx) * 0.5
                    } else {
                      const leftPx = u.valToPos(binCenter, "x", true)
                      const rightPx = u.valToPos(nextCenter, "x", true)
                      barWidth = Math.abs(rightPx - leftPx) * 0.5
                    }
                  }
                } else if (i > 0) {
                  const prevCenter = xData[i - 1]
                  if (Number.isFinite(prevCenter) && prevCenter > 0 && binCenter > 0) {
                    const leftPx = u.valToPos(prevCenter, "x", true)
                    const rightPx = u.valToPos(binCenter, "x", true)
                    barWidth = Math.abs(rightPx - leftPx) * 0.5
                  }
                }
              }

              // Final fallback if still no width - use average spacing
              if (barWidth <= 0 && xData.length > 0) {
                const xMin = u.scales.x.min || (xScaleType === "log" ? 0.1 : 0)
                const xMax = u.scales.x.max || 1
                const totalWidthPx = width * 0.8
                barWidth = Math.max(2, totalWidthPx / xData.length)
              }

              const xPx = u.valToPos(binCenter, "x", true)
              const yPx = u.valToPos(count, "y", true)
              const zeroPx = u.valToPos(0, "y", true)

              const barHeight = Math.abs(zeroPx - yPx)
              const barLeft = xPx - barWidth / 2
              const barTop = Math.min(zeroPx, yPx)

              // Draw filled rectangle
              if (barWidth > 0 && barHeight > 0) {
                ctx.fillRect(barLeft, barTop, barWidth, barHeight)
                ctx.strokeRect(barLeft, barTop, barWidth, barHeight)
              }
            }

            ctx.restore()
          }
        ]
      }
    }
  }, [width, height, isDark, data, xLabel, yLabel, barColor, edgeColor, xScaleType])

  const chartData = useMemo(() => {
    const xData = data.bins || []
    const yData = data.counts || []
    // Return data in uPlot format: [xData, yData]
    // Ensure arrays are properly typed
    return [xData as number[], yData as number[]]
  }, [data])

  // Force uPlot to update scale when data changes
  useEffect(() => {
    if (uPlotInstanceRef.current && data.bins && data.bins.length > 0) {
      const u = uPlotInstanceRef.current
      const xData = data.bins || []
      const binEdges = data.bin_edges || []
      
      // Recalculate and set x scale
      if (binEdges.length >= 2) {
        const validEdges = binEdges.filter(v => Number.isFinite(v) && (xScaleType === "linear" || v > 0))
        if (validEdges.length >= 2) {
          const minVal = Math.min(...validEdges)
          const maxVal = Math.max(...validEdges)
          if (xScaleType === "linear") {
            const range = maxVal - minVal
            const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
            u.setScale("x", { 
              min: Math.max(0, minVal - padding), 
              max: maxVal + padding 
            })
          } else if (xScaleType === "log" && minVal > 0) {
            const padding = 0.1
            u.setScale("x", { 
              min: minVal / (1 + padding), 
              max: maxVal * (1 + padding) 
            })
          }
        }
      } else if (xData.length > 0) {
        const validXData = xData.filter(v => Number.isFinite(v) && (xScaleType === "linear" || v > 0))
        if (validXData.length > 0) {
          const minVal = Math.min(...validXData)
          const maxVal = Math.max(...validXData)
          if (xScaleType === "linear") {
            const range = maxVal - minVal
            const padding = range > 0 ? range * 0.05 : Math.abs(maxVal) * 0.05 || 1
            u.setScale("x", { min: Math.max(0, minVal - padding), max: maxVal + padding })
          }
        }
      }
      
      // Set y scale
      const yData = data.counts || []
      if (yData.length > 0) {
        const maxCount = Math.max(...yData)
        u.setScale("y", { min: 0, max: maxCount * 1.1 })
      }
    }
  }, [data, xScaleType])

  // Check if we have valid data
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
      <UplotReact options={opts} data={chartData} />
    </div>
  )
}

