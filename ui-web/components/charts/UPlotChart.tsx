"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "next-themes"
import uPlot from "uplot"

const UplotReact = dynamic(() => import("react-uplot").then((mod) => mod.UPlot), { ssr: false })

type NumberArray = number[] | Float32Array | Float64Array

export type Series = {
  label: string
  color: string
  width?: number
  data: NumberArray
  points?: boolean
  pointSize?: number
	dash?: number[]
}

interface UPlotChartProps {
  xData: NumberArray
  series: Series[]
  xLabel?: string
  yLabel?: string
  className?: string
  height?: number
  onResetZoom?: () => void
  widthSegments?: { x0: number; x1: number; y: number }[]
	hLines?: { y: number; color?: string; dash?: number[] }[]
	vLines?: { x: number; color?: string; dash?: number[] }[]
	xRange?: { min: number; max: number } | null
	onXRangeChange?: (range: { min: number; max: number }) => void
	xScaleType?: "linear" | "log"
	yScaleType?: "linear" | "log"
	yRange?: { min: number; max: number } | null
	legend?: boolean
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

export function UPlotChart({
  xData,
  series,
  xLabel = "Time (min)",
  yLabel = "Counts",
  className = "",
  height = 400,
  onResetZoom,
  widthSegments,
	hLines,
	vLines,
	xRange,
	onXRangeChange,
	xScaleType = "linear",
	yScaleType = "linear",
	yRange,
	legend = true,
}: UPlotChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const uPlotInstanceRef = useRef<uPlot | null>(null)
  const [width, setWidth] = useState<number>(800)
  const [mounted, setMounted] = useState(false)
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === "dark"

  // Ensure component is mounted before rendering UPlot
  useEffect(() => {
    setMounted(true)
  }, [])

  // Handle responsive width
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const w = entries[0].contentRect.width
      setWidth(Math.max(300, Math.floor(w)))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const opts: uPlot.Options = useMemo(() => {
    const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)"
    const axisColor = isDark ? "#e5e7eb" : "#1f2937"

    return {
      width,
      height,
      scales: {
				x: { 
					time: false,
					...(xScaleType === "log" ? { distr: 2 as const } : {}),
					...(xRange ? { min: xRange.min, max: xRange.max } : {}),
				},
				y: { 
					auto: true,
					...(yScaleType === "log" ? { distr: 2 as const } : {}),
					...(yRange ? { min: yRange.min, max: yRange.max } : {}),
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
				show: !!legend,
      },
      series: [
        {}, // x axis
        ...series.map(s => ({
          label: s.label,
          stroke: s.color,
          width: s.width ?? 1.5,
					dash: s.dash,
          points: s.points ? {
            show: true,
            size: s.pointSize ?? 5,
            stroke: s.color,
            fill: s.color,
          } : undefined,
        })),
      ],
      hooks: {
				setScale: [
					(u: uPlot, key: string) => {
						if (key === "x" && onXRangeChange) {
							const min = u.scales.x.min!
							const max = u.scales.x.max!
							if (Number.isFinite(min) && Number.isFinite(max)) {
								onXRangeChange({ min, max })
							}
						}
					}
				],
        ready: [
          (u: uPlot) => {
            uPlotInstanceRef.current = u
            const over = u.over

            // Wheel zoom
            over.addEventListener("wheel", (e: WheelEvent) => {
              e.preventDefault()
              const rect = over.getBoundingClientRect()
              const leftPx = e.clientX - rect.left
              const xVal = u.posToVal(leftPx, "x")
              const factor = e.deltaY < 0 ? 0.75 : 1 / 0.75
              const min = u.scales.x.min!
              const max = u.scales.x.max!
              const nxMin = xVal - (xVal - min) * factor
              const nxMax = xVal + (max - xVal) * factor
              u.setScale("x", { min: nxMin, max: nxMax })
            }, { passive: false })

            // Double-click to reset zoom
            over.addEventListener("dblclick", () => {
              if (onResetZoom) {
                onResetZoom()
              } else {
                u.setScale("x", { min: xData[0], max: xData[xData.length - 1] })
                const baseY = (series && series.length > 0 ? series[0].data : []) as number[]
                const finiteY = baseY.filter(v => typeof v === "number" && Number.isFinite(v))
                if (finiteY.length > 0) {
                  const yMin = Math.min(...finiteY)
                  const yMax = Math.max(...finiteY)
                  u.setScale("y", { min: yMin, max: yMax })
                }
              }
            })

            over.style.cursor = "crosshair"
          }
        ]
        ,
        draw: [
          (u: uPlot) => {
						const ctx = u.ctx
						// Draw width segments (horizontal segments at given y between x0-x1)
						if (widthSegments && widthSegments.length > 0) {
							ctx.save()
							ctx.lineWidth = 2
							ctx.strokeStyle = isDark ? "#f59e0b" : "#d97706"
							widthSegments.forEach(seg => {
								const x0px = Math.round(u.valToPos(seg.x0, "x", true))
								const x1px = Math.round(u.valToPos(seg.x1, "x", true))
								const ypx = Math.round(u.valToPos(seg.y, "y", true))
								ctx.beginPath()
								ctx.moveTo(x0px, ypx)
								ctx.lineTo(x1px, ypx)
								ctx.stroke()
							})
							ctx.restore()
						}

						// Draw horizontal threshold lines across full plot
						if (hLines && hLines.length > 0) {
							ctx.save()
							hLines.forEach(line => {
								const ypx = Math.round(u.valToPos(line.y, "y", true))
								const x0px = Math.round(u.valToPos(u.scales.x.min!, "x", true))
								const x1px = Math.round(u.valToPos(u.scales.x.max!, "x", true))
								ctx.beginPath()
								ctx.setLineDash(line.dash ?? [6, 6])
								ctx.lineWidth = 1.5
								ctx.strokeStyle = line.color ?? (isDark ? "#f97316" : "#ea580c")
								ctx.moveTo(x0px, ypx)
								ctx.lineTo(x1px, ypx)
								ctx.stroke()
							})
							ctx.restore()
						}

						// Draw vertical threshold lines across full plot
						if (vLines && vLines.length > 0) {
							ctx.save()
							vLines.forEach(line => {
								const xpx = Math.round(u.valToPos(line.x, "x", true))
								const y0px = Math.round(u.valToPos(u.scales.y.min!, "y", true))
								const y1px = Math.round(u.valToPos(u.scales.y.max!, "y", true))
								ctx.beginPath()
								ctx.setLineDash(line.dash ?? [6, 6])
								ctx.lineWidth = 1.5
								ctx.strokeStyle = line.color ?? (isDark ? "#f97316" : "#ea580c")
								ctx.moveTo(xpx, y0px)
								ctx.lineTo(xpx, y1px)
								ctx.stroke()
							})
							ctx.restore()
						}
          }
        ]
      }
		}
	}, [width, height, isDark, series, xLabel, yLabel, xData, onResetZoom, widthSegments, hLines, vLines, xRange, onXRangeChange, xScaleType, yScaleType, legend])

  const data = useMemo(() => {
    const result: any[] = [xData]
    series.forEach(s => result.push(s.data))
    return result
  }, [xData, series])

  // Preserve zoom when data updates
  useEffect(() => {
    if (uPlotInstanceRef.current && xRange) {
      // When data changes, preserve the zoom range
      const u = uPlotInstanceRef.current
      const currentMin = u.scales.x.min
      const currentMax = u.scales.x.max
      
      // Only update if the range is different (avoid feedback loop)
      if (currentMin !== xRange.min || currentMax !== xRange.max) {
        u.setScale("x", { min: xRange.min, max: xRange.max })
      }
    }
  }, [data, xRange]) // Update when data or xRange changes

  if (!xData.length || !series.length) {
    return (
      <div ref={containerRef} className={`flex items-center justify-center ${className}`} style={{ height }}>
        <p className="text-sm text-muted-foreground">No data available</p>
      </div>
    )
  }

  // Wait for client-side mount before rendering UPlot
  if (!mounted) {
    return (
      <div ref={containerRef} className={className} style={{ height, minHeight: 0 }} />
    )
  }

  return (
    <div ref={containerRef} className={className} style={{ minHeight: 0 }}>
      <UplotReact options={opts} data={data} />
    </div>
  )
}

