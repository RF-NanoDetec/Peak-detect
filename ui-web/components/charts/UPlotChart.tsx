"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "@/hooks/use-theme"
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
  bar?: boolean
  barWidth?: number
}

export type WidthSegment = { 
  x0: number
  x1: number
  y: number
  width?: number  // Optional: the actual width value for tooltip display
}

interface UPlotChartProps {
  xData: NumberArray
  series: Series[]
  xLabel?: string
  yLabel?: string
  className?: string
  height?: number
  onResetZoom?: () => void
  widthSegments?: WidthSegment[]
	hLines?: { y: number; color?: string; dash?: number[]; label?: string }[]
	vLines?: { x: number; color?: string; dash?: number[]; label?: string }[]
	xRange?: { min: number; max: number } | null
	onXRangeChange?: (range: { min: number; max: number }) => void
	xScaleType?: "linear" | "log"
	yScaleType?: "linear" | "log"
	yRange?: { min: number; max: number } | null
	legend?: boolean
	enableYAxisZoom?: boolean
	enableLasso?: boolean
	onLassoComplete?: (selectedIndices: number[]) => void
	selectedIndices?: number[]
	pointColors?: string[] // Per-point colors for scatter plots
	filteredIndices?: number[] // Indices that pass filters (for dimming others)
	customScatter?: boolean // Enable custom scatter rendering (Double Peak only)
}

const formatAxisNumber = (value: number | null | undefined): string => {
  if (value == null || Number.isNaN(value) || !Number.isFinite(value)) return ""
  if (value === 0) return "0"
  const abs = Math.abs(value)
  if (abs < 0.001 || abs > 10000) return value.toExponential(1)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  if (abs >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

// Point-in-polygon test using ray casting algorithm
function isPointInPolygon(x: number, y: number, polygon: { x: number; y: number }[]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x
    const yi = polygon[i].y
    const xj = polygon[j].x
    const yj = polygon[j].y
    
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
    if (intersect) inside = !inside
  }
  return inside
}

type TimeUnit = "min" | "s" | "ms"

// Format x-axis ticks for time (input values are minutes)
const formatTimeAxisTicks = (
  u: uPlot, 
  vals: number[], 
  offsetInfoRef: React.MutableRefObject<{ offset: number; unit: TimeUnit; decimals?: number } | null>
): string[] => {
  const xMin = u.scales.x.min ?? (vals.length ? vals[0] : 0)
  const xMax = u.scales.x.max ?? (vals.length ? vals[vals.length - 1] : xMin)
  const spanMin = Math.max(0, (xMax ?? 0) - (xMin ?? 0))
  const spanSec = spanMin * 60 // Convert span to seconds
  const stepMin = vals.length >= 2
    ? Math.max(0, vals[1] - vals[0])
    : (spanMin > 0 ? spanMin / Math.max(vals.length - 1, 1) : 0)
  const stepSec = stepMin * 60

  let unit: TimeUnit
  if (spanSec < 1) {
    // If visible range is less than 1 second, use milliseconds
    unit = "ms"
  } else if (spanMin < 1) {
    // If visible range is less than 1 minute but >= 1 second, use seconds
    unit = "s"
  } else {
    unit = "min"
  }

  // Convert values to display unit and check if any exceed 100
  const convertedVals = vals.map(v => {
    if (!Number.isFinite(v)) return NaN
    if (unit === "min") return v
    if (unit === "s") return v * 60
    return v * 60000 // ms
  })
  
  const finiteVals = convertedVals.filter(v => Number.isFinite(v))
  const minVal = finiteVals.length > 0 ? Math.min(...finiteVals) : 0
  const maxVal = finiteVals.length > 0 ? Math.max(...finiteVals) : 0
  
  // For milliseconds, determine decimal places based on step size to avoid duplicates
  const stepMs = stepMin * 60000
  let msDecimals = 0
  if (unit === "ms") {
    if (stepMs < 0.01) {
      msDecimals = 3  // 0.001 ms precision
    } else if (stepMs < 0.1) {
      msDecimals = 2  // 0.01 ms precision
    } else if (stepMs < 1) {
      msDecimals = 1  // 0.1 ms precision
    } else {
      msDecimals = 0  // Integer ms
    }
  }
  
  // Use relative display if any value exceeds 100
  const useRelative = maxVal > 100 || minVal > 100
  
  if (useRelative) {
    // Store offset info for annotation (include decimals for ms formatting)
    offsetInfoRef.current = { 
      offset: minVal, 
      unit,
      decimals: unit === "ms" ? msDecimals : undefined
    }
    
    // Return relative values
    if (unit === "min") {
      const decimals =
        stepMin >= 10 ? 0 :
        stepMin >= 1 ? 0 :
        stepMin >= 0.1 ? 1 : 2
      return vals.map(v => {
        if (!Number.isFinite(v)) return ""
        const relative = v - minVal
        return relative.toFixed(decimals)
      })
    }
    
    if (unit === "s") {
      const decimals = stepSec >= 5 ? 0 : 1
      return vals.map(v => {
        if (!Number.isFinite(v)) return ""
        const s = v * 60
        const relative = s - minVal
        return relative.toFixed(decimals)
      })
    }
    
    // ms
    return vals.map(v => {
      if (!Number.isFinite(v)) return ""
      const ms = v * 60000
      const relative = ms - minVal
      return relative.toFixed(msDecimals)
    })
  } else {
    // Clear offset info when not using relative display
    offsetInfoRef.current = null
    
    // Original absolute display
    if (unit === "min") {
      const decimals =
        stepMin >= 10 ? 0 :
        stepMin >= 1 ? 0 :
        stepMin >= 0.1 ? 1 : 2
      return vals.map(v => (Number.isFinite(v) ? v.toFixed(decimals) : ""))
    }

    if (unit === "s") {
      const decimals = stepSec >= 5 ? 0 : 1
      return vals.map(v => {
        const s = v * 60
        return Number.isFinite(s) ? s.toFixed(decimals) : ""
      })
    }

    // ms - use appropriate decimal places based on step size
    return vals.map(v => {
      const ms = v * 60000
      return Number.isFinite(ms) ? ms.toFixed(msDecimals) : ""
    })
  }
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
	enableYAxisZoom = false,
	enableLasso = false,
	onLassoComplete,
	selectedIndices = [],
	pointColors,
	filteredIndices,
	customScatter = false,
}: UPlotChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const uPlotInstanceRef = useRef<uPlot | null>(null)
  const offsetInfoRef = useRef<{ offset: number; unit: TimeUnit; decimals?: number } | null>(null)
  const [width, setWidth] = useState<number>(800)
  const [mounted, setMounted] = useState(false)
  const [hoveredSegment, setHoveredSegment] = useState<{ index: number; width: number } | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  
  // Refs for lasso state to avoid re-renders/re-initialization
  const lassoPathRef = useRef<{ x: number; y: number }[]>([])
  const isDrawingLassoRef = useRef(false)
  
  // Refs for props to access in uPlot hooks without recreating options
  const selectedIndicesRef = useRef(selectedIndices)
  const filteredIndicesRef = useRef(filteredIndices)
  const enableLassoRef = useRef(enableLasso)
  
  // Cache Sets for performance in draw loop
  const filteredSetRef = useRef<Set<number> | null>(null)
  const selectedSetRef = useRef<Set<number> | null>(null)

  const { theme } = useTheme()
  const isDark = theme === "dark"

  // Sync refs with props and update cached Sets
  selectedIndicesRef.current = selectedIndices
  filteredIndicesRef.current = filteredIndices
  enableLassoRef.current = enableLasso
  
  // Update Sets when arrays change (shallow comparison is enough if references change)
  useMemo(() => {
    filteredSetRef.current = filteredIndices ? new Set(filteredIndices) : null
  }, [filteredIndices])
  
  useMemo(() => {
    selectedSetRef.current = selectedIndices && selectedIndices.length > 0 ? new Set(selectedIndices) : null
  }, [selectedIndices])

  // Apply y-range updates immediately when provided, then redraw
  useEffect(() => {
    if (uPlotInstanceRef.current && yRange) {
      const u = uPlotInstanceRef.current
      const min = Number.isFinite(yRange.min) ? yRange.min : (u.scales.y.min ?? 0)
      const max = Number.isFinite(yRange.max) ? yRange.max : (u.scales.y.max ?? 1)
      if (Number.isFinite(min) && Number.isFinite(max) && max > min) {
        u.setScale("y", { min, max })
        // Redraw after scale update to ensure lines are positioned correctly
        requestAnimationFrame(() => {
          if (uPlotInstanceRef.current) {
            uPlotInstanceRef.current.redraw()
          }
        })
      }
    }
  }, [yRange])

  // Force redraw on dynamic overlay changes (lines, selections, filters)
  // Use requestAnimationFrame to defer redraw until after any scale updates
  useEffect(() => {
    if (uPlotInstanceRef.current) {
      requestAnimationFrame(() => {
        if (uPlotInstanceRef.current) {
          // Redraw without recalculating scales to preserve zoom
          uPlotInstanceRef.current.redraw(false, false)
        }
      })
    }
  }, [hLines, vLines, selectedIndices, filteredIndices])

  // Lasso selection: attach/detach pointer listeners when enableLasso changes
  useEffect(() => {
    const u = uPlotInstanceRef.current
    if (!u || !onLassoComplete) return

    const over = u.over
    if (!over) return

    if (!enableLasso) {
      isDrawingLassoRef.current = false
      lassoPathRef.current = []
      u.redraw(false, false)
      return
    }

    let drawing = false
    let pointerId: number | null = null
    let lassoPathLocal: { x: number; y: number }[] = []

    const getLocalCoords = (clientX: number, clientY: number) => {
      const canvas = u.ctx.canvas
      const canvasRect = canvas.getBoundingClientRect()
      const transform = u.ctx.getTransform()
      
      // Calculate scaling factors between physical pixels and CSS pixels
      // This handles high DPI displays and browser zoom levels
      const pixelRatioX = canvas.width / canvasRect.width
      const pixelRatioY = canvas.height / canvasRect.height
      
      // Adjust for any transform applied to the context (uPlot usually scales by dpr)
      const scaleX = transform.a
      const scaleY = transform.d
      
      return {
        x: (clientX - canvasRect.left) * pixelRatioX / scaleX,
        y: (clientY - canvasRect.top) * pixelRatioY / scaleY,
      }
    }

    const completeSelection = () => {
      if (lassoPathLocal.length > 2) {
        const selected: number[] = []
        const firstSeries = series.find(s => s.points)
        if (firstSeries) {
          for (let i = 0; i < xData.length; i++) {
            const xVal = xData[i]
            const yVal = firstSeries.data[i]
            if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue

            const xPx = u.valToPos(xVal, "x", true)
            const yPx = u.valToPos(yVal as number, "y", true)

            if (isPointInPolygon(xPx, yPx, lassoPathLocal)) {
              selected.push(i)
            }
          }
        }
        onLassoComplete(selected)
      } else {
        onLassoComplete([])
      }
    }

    const resetLassoState = () => {
      drawing = false
      lassoPathLocal = []
      pointerId = null
      isDrawingLassoRef.current = false
      lassoPathRef.current = []
      u.redraw(false, false)
    }

    const handlePointerDown = (e: PointerEvent) => {
      if (!enableLassoRef.current || e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()

      drawing = true
      pointerId = e.pointerId
      lassoPathLocal = [getLocalCoords(e.clientX, e.clientY)]
      
      isDrawingLassoRef.current = true
      lassoPathRef.current = [...lassoPathLocal]
      u.redraw(false, false)

      try {
        over.setPointerCapture(e.pointerId)
      } catch {
        // ignore if capture fails
      }
    }

    const handlePointerMove = (e: PointerEvent) => {
      if (!drawing || pointerId !== e.pointerId) return
      e.preventDefault()
      const point = getLocalCoords(e.clientX, e.clientY)
      lassoPathLocal.push(point)
      
      lassoPathRef.current = [...lassoPathLocal]
      u.redraw(false, false)
    }

    const handlePointerUp = (e: PointerEvent) => {
      if (!drawing || pointerId !== e.pointerId) return
      e.preventDefault()
      completeSelection()
      resetLassoState()
      try {
        over.releasePointerCapture(e.pointerId)
      } catch {
        // ignore
      }
    }

    const handlePointerCancel = (e: PointerEvent) => {
      if (!drawing || pointerId !== e.pointerId) return
      completeSelection()
      resetLassoState()
    }

    const handleContextMenu = (e: MouseEvent) => {
      if (enableLassoRef.current) {
        e.preventDefault()
      }
    }

    over.addEventListener("pointerdown", handlePointerDown)
    window.addEventListener("pointermove", handlePointerMove)
    window.addEventListener("pointerup", handlePointerUp)
    window.addEventListener("pointercancel", handlePointerCancel)
    over.addEventListener("contextmenu", handleContextMenu)
    over.style.cursor = "crosshair"

    return () => {
      over.removeEventListener("pointerdown", handlePointerDown)
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
      window.removeEventListener("pointercancel", handlePointerCancel)
      over.removeEventListener("contextmenu", handleContextMenu)
      
      isDrawingLassoRef.current = false
      lassoPathRef.current = []
      
      try {
        if (pointerId !== null) {
          over.releasePointerCapture(pointerId)
        }
      } catch {
        // ignore
      }
    }
  }, [enableLasso, onLassoComplete, xData, series])

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
    const gridColor = isDark ? "rgba(248,250,252,0.10)" : "rgba(15,23,42,0.06)"
    const axisColor = isDark ? "rgba(248,250,252,0.88)" : "rgba(15,23,42,0.65)"
    const axisFont = "600 11px 'Inter', 'Segoe UI', system-ui, sans-serif"
    const axisLabelFont = "600 12px 'Inter', 'Segoe UI', system-ui, sans-serif"

    // Dynamic x-axis label unit based on current visible span (xData is in minutes)
    const currentXMin = (xRange?.min ?? (xData.length ? xData[0] : 0))
    const currentXMax = (xRange?.max ?? (xData.length ? xData[xData.length - 1] : 0))
    const visibleSpanMin = Math.max(0, currentXMax - currentXMin)
    const visibleSpanSec = visibleSpanMin * 60
    let timeUnitLabel: TimeUnit = "min"
    if (visibleSpanSec < 1) {
      // If visible range is less than 1 second, use milliseconds
      timeUnitLabel = "ms"
    } else if (visibleSpanMin < 1) {
      // If visible range is less than 1 minute but >= 1 second, use seconds
      timeUnitLabel = "s"
    } else {
      timeUnitLabel = "min"
    }
    const baseXLabel = (xLabel || "Time").replace(/\s*\(.*\)\s*$/, "")
    const dynamicXLabel = `${baseXLabel} (${timeUnitLabel})`

    const yLogConfig: Partial<uPlot.Scale> =
      yScaleType === "log"
        ? {
            distr: 3 as const,
            log: 10,
            clamp: (_self: uPlot, val: number) => Math.max(val, 1e-9),
            range: (_u: uPlot, dataMin: number, dataMax: number) => {
              let min = yRange?.min ?? dataMin
              let max = yRange?.max ?? dataMax

              if (!Number.isFinite(min) || min <= 0) {
                min = 1e-3
              }
              if (!Number.isFinite(max) || max <= min) {
                max = min * 10
              }
              return [min, max]
            },
          }
        : {}

    return {
      width,
      height,
      cursor: {
        drag: {
          x: false,
          y: false,
        },
        sync: {
          key: Math.random().toString(),
        },
      },
      scales: {
				x: { 
					time: false,
					...(xScaleType === "log"
            ? {
                distr: 3 as const,
                log: 10,
                clamp: (_self: uPlot, val: number) => Math.max(val, 1e-9),
              }
            : {}),
					...(xRange ? { min: xRange.min, max: xRange.max } : {}),
				},
				y: { 
					...(yRange ? { auto: false, min: yRange.min, max: yRange.max } : { auto: true }),
          ...yLogConfig,
				},
      },
      axes: [
        {
          stroke: axisColor,
          grid: { stroke: gridColor, width: 1 },
          label: dynamicXLabel,
          labelSize: 20,
          font: axisFont,
          labelFont: axisLabelFont,
          size: 50,
          values: (u: uPlot, vals: number[]) => formatTimeAxisTicks(u, vals, offsetInfoRef),
        },
        {
          stroke: axisColor,
          grid: { stroke: gridColor, width: 1 },
          label: yLabel,
          labelSize: 30,
          font: axisFont,
          labelFont: axisLabelFont,
          size: 60,
          values: (u: uPlot, vals: number[]) => vals.map(formatAxisNumber),
        },
      ],
      legend: {
				show: !!legend,
        stroke: isDark ? "rgba(248,250,252,0.95)" : "#1f2937",
        fill: isDark ? "rgba(0,0,0,0)" : "rgba(255,255,255,0)",
      },
      series: [
        {}, // x axis
        ...series.map(s => ({
          label: s.label,
          stroke: s.color,
          // Only suppress lines when custom scatter styling is enabled for this chart
          width: s.bar ? 0 : (customScatter && s.points ? 0 : (s.width ?? 1.5)),
					dash: s.dash,
          points: s.points ? {
            // Hide default points only when we render custom ones ourselves
            show: customScatter ? false : true,
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
            
            // Style legend text for dark mode
            const legendEl = u.root.querySelector('.u-legend')
            if (legendEl) {
              const legendTextColor = isDark ? "rgba(248,250,252,0.95)" : "#1f2937"
              ;(legendEl as HTMLElement).style.color = legendTextColor
              // Also style individual legend items
              const legendItems = legendEl.querySelectorAll('.u-legend-item')
              legendItems.forEach((item) => {
                ;(item as HTMLElement).style.color = legendTextColor
              })
            }

            // Wheel zoom (disabled when lasso is active)
            over.addEventListener("wheel", (e: WheelEvent) => {
              // Don't zoom when lasso is active
              if (enableLassoRef.current) return
              
              e.preventDefault()
              const rect = over.getBoundingClientRect()
              const leftPx = e.clientX - rect.left
              const topPx = e.clientY - rect.top
              
              if (enableYAxisZoom && e.shiftKey) {
                // Shift+scroll: zoom y-axis
                const yVal = u.posToVal(topPx, "y")
                const factor = e.deltaY < 0 ? 0.75 : 1 / 0.75
                const min = u.scales.y.min!
                const max = u.scales.y.max!
                const nyMin = yVal - (yVal - min) * factor
                const nyMax = yVal + (max - yVal) * factor
                u.setScale("y", { min: nyMin, max: nyMax })
              } else if (enableYAxisZoom) {
                // When y-axis zoom is enabled, zoom the axis closer to the mouse
                const plotLeft = u.bbox.left
                const plotTop = u.bbox.top
                const plotWidth = u.bbox.width
                const plotHeight = u.bbox.height
                const distToLeft = leftPx - plotLeft
                const distToRight = (plotLeft + plotWidth) - leftPx
                const distToTop = topPx - plotTop
                const distToBottom = (plotTop + plotHeight) - topPx
                
                const minDistX = Math.min(distToLeft, distToRight)
                const minDistY = Math.min(distToTop, distToBottom)
                
                if (minDistY < minDistX) {
                  // Zoom y-axis
                  const yVal = u.posToVal(topPx, "y")
                  const factor = e.deltaY < 0 ? 0.75 : 1 / 0.75
                  const min = u.scales.y.min!
                  const max = u.scales.y.max!
                  const nyMin = yVal - (yVal - min) * factor
                  const nyMax = yVal + (max - yVal) * factor
                  u.setScale("y", { min: nyMin, max: nyMax })
                } else {
                  // Zoom x-axis
                  const xVal = u.posToVal(leftPx, "x")
                  const factor = e.deltaY < 0 ? 0.75 : 1 / 0.75
                  const min = u.scales.x.min!
                  const max = u.scales.x.max!
                  const nxMin = xVal - (xVal - min) * factor
                  const nxMax = xVal + (max - xVal) * factor
                  u.setScale("x", { min: nxMin, max: nxMax })
                }
              } else {
                // Default: zoom x-axis only
                const xVal = u.posToVal(leftPx, "x")
                const factor = e.deltaY < 0 ? 0.75 : 1 / 0.75
                const min = u.scales.x.min!
                const max = u.scales.x.max!
                const nxMin = xVal - (xVal - min) * factor
                const nxMax = xVal + (max - xVal) * factor
                u.setScale("x", { min: nxMin, max: nxMax })
              }
            }, { passive: false })

            // Double-click to reset zoom (disabled when lasso is active)
            over.addEventListener("dblclick", (e: MouseEvent) => {
              // Don't reset zoom when lasso is active
              if (enableLassoRef.current) return
              
              e.preventDefault()
              e.stopPropagation()
              
              if (onResetZoom) {
                onResetZoom()
              } else {
                // Calculate full data range for x-axis
                const xMin = xData.length > 0 ? xData[0] : 0
                const xMax = xData.length > 0 ? xData[xData.length - 1] : 1
                
                // If xRange is controlled, update parent state first to trigger re-render
                if (onXRangeChange) {
                  onXRangeChange({ min: xMin, max: xMax })
                }
                
                // Reset x-axis
                u.setScale("x", { min: xMin, max: xMax })
                
                // Reset y-axis - gather all finite values from all series
                const allYValues: number[] = []
                series.forEach(s => {
                  const yVals = Array.from(s.data).filter(v => typeof v === "number" && Number.isFinite(v)) as number[]
                  allYValues.push(...yVals)
                })
                
                if (allYValues.length > 0) {
                  const yMin = Math.min(...allYValues)
                  const yMax = Math.max(...allYValues)
                  u.setScale("y", { min: yMin, max: yMax })
                }
              }
            })

            // Mouse move for width segment hover detection (disabled when lasso is active)
            over.addEventListener("mousemove", (e: MouseEvent) => {
              // Don't show hover tooltips when lasso is active
              if (enableLassoRef.current) {
                setHoveredSegment(null)
                return
              }
              
              if (!widthSegments || widthSegments.length === 0) {
                setHoveredSegment(null)
                return
              }
              
              const overRect = over.getBoundingClientRect()
              const containerRect = containerRef.current?.getBoundingClientRect()
              const mouseX = e.clientX - overRect.left
              const mouseY = e.clientY - overRect.top

              // Position relative to the React container so the tooltip follows correctly
              const containerX = containerRect ? e.clientX - containerRect.left : mouseX
              const containerY = containerRect ? e.clientY - containerRect.top : mouseY
              
              // Store mouse position for tooltip (container-relative)
              setMousePos({ x: containerX, y: containerY })
              
              // Check each width segment
              let found = false
              for (let i = 0; i < widthSegments.length; i += 1) {
                const seg = widthSegments[i]
                const x0px = u.valToPos(seg.x0, "x", true)
                const x1px = u.valToPos(seg.x1, "x", true)
                const ypx = u.valToPos(seg.y, "y", true)
                
                // Generous hitbox around the segment
                const hoverThresholdY = 8
                const hoverPaddingX = 5
                if (
                  mouseX >= x0px - hoverPaddingX &&
                  mouseX <= x1px + hoverPaddingX &&
                  Math.abs(mouseY - ypx) <= hoverThresholdY
                ) {
                  // Calculate width: either from seg.width or from x1-x0
                  const widthValue = seg.width ?? (seg.x1 - seg.x0)
                  setHoveredSegment({ index: i, width: widthValue })
                  found = true
                  break
                }
              }
              
              if (!found) {
                setHoveredSegment(null)
              }
            })

            // Mouse leave to clear hover
            over.addEventListener("mouseleave", () => {
              setHoveredSegment(null)
              setMousePos(null)
            })

            // Set initial cursor
            over.style.cursor = "crosshair"
          }
        ]
        ,
        draw: [
          (u: uPlot) => {
						const ctx = u.ctx
						
						// Update legend text color for dark mode (runs on every draw to catch theme changes)
						const legendEl = u.root.querySelector('.u-legend')
						if (legendEl) {
							const legendTextColor = isDark ? "rgba(248,250,252,0.95)" : "#1f2937"
							;(legendEl as HTMLElement).style.color = legendTextColor
							const legendItems = legendEl.querySelectorAll('.u-legend-item')
							legendItems.forEach((item) => {
								;(item as HTMLElement).style.color = legendTextColor
							})
						}

            // Draw bar-style series (e.g., throughput histogram over time)
            if (series && series.length > 0) {
              const xVals = (u.data[0] as number[]) || []
              series.forEach((s, sIdx) => {
                if (!s.bar) return
                const yVals = (u.data[sIdx + 1] as (number | null | undefined)[]) || []
                if (!xVals.length || !yVals.length) return

                ctx.save()
                ctx.fillStyle = s.color
                ctx.strokeStyle = s.color
                ctx.lineWidth = 1

                const baseY = yScaleType === "log" ? Math.max(u.scales.y.min ?? 1e-9, 1e-9) : (u.scales.y.min ?? 0)
                const baseYPx = u.valToPos(baseY, "y", true)

                // Compute bar width in pixels once, assuming roughly uniform spacing
                let barHalfWidthPx = 0
                if (s.barWidth && s.barWidth > 0) {
                  const leftVal = xVals[0] - s.barWidth / 2
                  const rightVal = xVals[0] + s.barWidth / 2
                  barHalfWidthPx = Math.abs(u.valToPos(rightVal, "x", true) - u.valToPos(leftVal, "x", true)) / 2
                } else if (xVals.length > 1) {
                  const dx = Math.abs(xVals[1] - xVals[0])
                  const leftVal = xVals[0] - dx / 2
                  const rightVal = xVals[0] + dx / 2
                  barHalfWidthPx = Math.abs(u.valToPos(rightVal, "x", true) - u.valToPos(leftVal, "x", true)) / 2
                } else {
                  barHalfWidthPx = Math.max(2, width * 0.02)
                }

                for (let i = 0; i < xVals.length && i < yVals.length; i += 1) {
                  const xv = xVals[i]
                  const yv = yVals[i]
                  if (yv == null || !Number.isFinite(yv)) continue

                  const xPx = u.valToPos(xv, "x", true)
                  const yPx = u.valToPos(
                    yScaleType === "log" ? Math.max(yv, 1e-9) : yv,
                    "y",
                    true,
                  )

                  const barLeft = xPx - barHalfWidthPx
                  const barRight = xPx + barHalfWidthPx
                  const barTop = Math.min(baseYPx, yPx)
                  const barHeight = Math.abs(baseYPx - yPx)

                  if (barHeight <= 0) continue

                  ctx.beginPath()
                  ctx.rect(barLeft, barTop, barRight - barLeft, barHeight)
                  ctx.fill()
                }

                ctx.restore()
              })
            }

						// Draw width segments (horizontal segments at given y between x0-x1 with vertical caps)
						if (widthSegments && widthSegments.length > 0) {
							ctx.save()
							const capHeight = 8 // Height of vertical caps in pixels
							
							widthSegments.forEach((seg, idx) => {
								const x0px = Math.round(u.valToPos(seg.x0, "x", true))
								const x1px = Math.round(u.valToPos(seg.x1, "x", true))
								const ypx = Math.round(u.valToPos(seg.y, "y", true))
								
								// Check if this segment is hovered
								const isHovered = hoveredSegment?.index === idx
								
								// Set styling - highlight if hovered
								ctx.lineWidth = isHovered ? 3 : 2
								ctx.strokeStyle = isHovered 
									? (isDark ? "#fbbf24" : "#f59e0b")  // Brighter when hovered
									: (isDark ? "#f59e0b" : "#d97706")
								
								// Draw horizontal line
								ctx.beginPath()
								ctx.moveTo(x0px, ypx)
								ctx.lineTo(x1px, ypx)
								ctx.stroke()
								
								// Draw left vertical cap
								ctx.beginPath()
								ctx.moveTo(x0px, ypx - capHeight / 2)
								ctx.lineTo(x0px, ypx + capHeight / 2)
								ctx.stroke()
								
								// Draw right vertical cap
								ctx.beginPath()
								ctx.moveTo(x1px, ypx - capHeight / 2)
								ctx.lineTo(x1px, ypx + capHeight / 2)
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
								
								// Draw label if provided
								if (line.label) {
									ctx.save()
									ctx.setLineDash([])
									ctx.fillStyle = line.color ?? (isDark ? "#f97316" : "#ea580c")
									ctx.font = "600 10px 'Inter', 'Segoe UI', system-ui, sans-serif"
									ctx.textAlign = "right"
									ctx.textBaseline = "bottom"
									ctx.fillText(line.label, x1px - 5, ypx - 3)
									ctx.restore()
								}
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

						// Draw offset annotation when using relative display
						if (offsetInfoRef.current) {
							const { offset, unit, decimals } = offsetInfoRef.current
							ctx.save()
							
							// Format the offset value
							let offsetText = ""
							if (unit === "ms") {
								const msDecimals = decimals ?? 0
								offsetText = `+ ${offset.toFixed(msDecimals)} ms`
							} else if (unit === "s") {
								offsetText = `+ ${offset.toFixed(1)} s`
							} else {
								offsetText = `+ ${offset.toFixed(2)} min`
							}
							
							// Style the annotation
							ctx.font = "600 11px 'Inter', 'Segoe UI', system-ui, sans-serif"
							ctx.fillStyle = isDark ? "rgba(248,250,252,0.75)" : "rgba(15,23,42,0.60)"
							ctx.textAlign = "left"
							ctx.textBaseline = "bottom"
							
							// Position at left edge of plot area, just above the x-axis
							const plotLeft = u.bbox.left
							const plotBottom = u.bbox.top + u.bbox.height
							const xPos = plotLeft + 5
							const yPos = plotBottom - 3
							
							ctx.fillText(offsetText, xPos, yPos)
							ctx.restore()
						}

						// Draw lasso path if drawing
            const lassoPath = lassoPathRef.current
            const isDrawingLasso = isDrawingLassoRef.current
            
						if (lassoPath.length > 1) {
							ctx.save()
							ctx.strokeStyle = isDark ? "#60a5fa" : "#3b82f6"
							ctx.lineWidth = 2
							ctx.setLineDash([5, 3])
							ctx.beginPath()
							ctx.moveTo(lassoPath[0].x, lassoPath[0].y)
							for (let i = 1; i < lassoPath.length; i++) {
								ctx.lineTo(lassoPath[i].x, lassoPath[i].y)
							}
							if (isDrawingLasso) {
								ctx.stroke()
							} else {
								// Close the path when complete
								ctx.closePath()
								ctx.stroke()
							}
							ctx.restore()
						}

						// Custom point rendering with three states (Double Peak only)
						const firstSeries = series.find(s => s.points)
						if (customScatter && firstSeries && xData.length > 0) {
							ctx.save()
                            // Clip to plot area to avoid drawing outside
                            ctx.beginPath();
                            ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
                            ctx.clip();
							
                            // Use cached sets for performance
                            const filteredSet = filteredSetRef.current
                            const selectedSet = selectedSetRef.current
                            
							const hasFilters = filteredSet && filteredSet.size < xData.length
                            
                            // Optimization: Group points by style to minimize state changes and draw calls
                            // 1. Selected (Light Blue, Large)
                            // 2. Normal (Default, Medium)
                            // 3. Filtered (Dim, Small)
                            
                            // We use path recording for batch rendering
                            const selectedPath = new Path2D();
                            const normalPath = new Path2D();
                            const filteredPath = new Path2D();
                            
                            let hasSelected = false;
                            let hasNormal = false;
                            let hasFiltered = false;
                            
                            // Viewport culling bounds
                            const xMin = u.bbox.left - 10; // Add margin for point radius
                            const xMax = u.bbox.left + u.bbox.width + 10;
                            const yMin = u.bbox.top - 10;
                            const yMax = u.bbox.top + u.bbox.height + 10;

							for (let i = 0; i < xData.length; i++) {
								const xVal = xData[i]
								const yVal = firstSeries.data[i]
								if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue
								
								const xPx = u.valToPos(xVal, "x", true)
								const yPx = u.valToPos(yVal as number, "y", true)
								
                                // Skip points outside viewport
                                if (xPx < xMin || xPx > xMax || yPx < yMin || yPx > yMax) continue;

								const isSelected = selectedSet?.has(i)
								const passesFilter = !hasFilters || filteredSet?.has(i)
                                
                                if (isSelected) {
                                    selectedPath.moveTo(xPx + 6, yPx);
                                    selectedPath.arc(xPx, yPx, 6, 0, 2 * Math.PI);
                                    hasSelected = true;
                                } else if (!passesFilter) {
                                    filteredPath.moveTo(xPx + 3, yPx);
                                    filteredPath.arc(xPx, yPx, 3, 0, 2 * Math.PI);
                                    hasFiltered = true;
                                } else {
                                    normalPath.moveTo(xPx + 4, yPx);
                                    normalPath.arc(xPx, yPx, 4, 0, 2 * Math.PI);
                                    hasNormal = true;
                                }
							}
							
                            // Draw batches
                            if (hasFiltered) {
                                ctx.fillStyle = isDark ? "rgba(100, 100, 100, 0.15)" : "rgba(150, 150, 150, 0.15)"
                                ctx.strokeStyle = isDark ? "rgba(100, 100, 100, 0.2)" : "rgba(150, 150, 150, 0.2)"
                                ctx.lineWidth = 0.5
                                ctx.fill(filteredPath)
                                ctx.stroke(filteredPath)
                            }
                            
                            if (hasNormal) {
                                ctx.fillStyle = isDark ? "rgba(203, 213, 225, 0.6)" : "rgba(15, 23, 42, 0.6)"
                                ctx.strokeStyle = isDark ? "rgba(203, 213, 225, 0.8)" : "rgba(15, 23, 42, 0.8)"
                                ctx.lineWidth = 1
                                ctx.fill(normalPath)
                                ctx.stroke(normalPath)
                            }
                            
                            if (hasSelected) {
                                ctx.fillStyle = isDark ? "rgba(96, 165, 250, 0.8)" : "rgba(59, 130, 246, 0.8)"
                                ctx.strokeStyle = isDark ? "#60a5fa" : "#3b82f6"
                                ctx.lineWidth = 2
                                ctx.fill(selectedPath)
                                ctx.stroke(selectedPath)
                            }
							
							ctx.restore()
						}
          }
        ]
      }
		}
	}, [width, height, isDark, series, xLabel, yLabel, xData, onResetZoom, widthSegments, hLines, vLines, xRange, onXRangeChange, xScaleType, yScaleType, yRange, legend, enableYAxisZoom])

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
    <div ref={containerRef} className={className} style={{ minHeight: 0, position: "relative" }}>
      <UplotReact options={opts} data={data} />
      
      {/* Tooltip for peak width */}
      {hoveredSegment && mousePos && (
        <div
          className="pointer-events-none absolute z-50 px-2 py-1 text-xs font-medium rounded shadow-lg border"
          style={{
            left: mousePos.x + 10,
            top: mousePos.y - 30,
            backgroundColor: isDark ? "rgba(30, 41, 59, 0.95)" : "rgba(255, 255, 255, 0.95)",
            color: isDark ? "rgba(248, 250, 252, 0.95)" : "rgba(15, 23, 42, 0.9)",
            borderColor: isDark ? "rgba(71, 85, 105, 0.5)" : "rgba(203, 213, 225, 0.8)",
          }}
        >
          <div className="whitespace-nowrap">
            Peak width (ms): {hoveredSegment.width.toFixed(3)}
          </div>
        </div>
      )}
    </div>
  )
}
