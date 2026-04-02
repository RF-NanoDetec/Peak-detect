"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "@/hooks/use-theme"
import uPlot from "uplot"
import { Camera } from "lucide-react"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

const UplotReact = dynamic(() => import("react-uplot").then((mod) => mod.UPlot), { ssr: false })

// Cleanup helper to properly destroy uPlot instance and prevent memory leaks
const cleanupUPlotInstance = (instanceRef: React.MutableRefObject<uPlot | null>) => {
  if (instanceRef.current) {
    try {
      instanceRef.current.destroy()
    } catch (e) {
      // Ignore errors during cleanup
    }
    instanceRef.current = null
  }
}

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
  dataIndices?: number[] // Optional: specific indices to render (optimization for sparse data)
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
  onYRangeChange?: (range: { min: number; max: number }) => void
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

// Safe log axis splits function that limits the number of splits to prevent RangeError
// uPlot's default logAxisSplits can create too many elements when the range spans many orders of magnitude
const safeLogAxisSplits = (u: uPlot, axisIdx: number, scaleMin: number, scaleMax: number, foundIncr: number, foundSpace: number): number[] => {
  // Ensure valid inputs
  if (!Number.isFinite(scaleMin) || !Number.isFinite(scaleMax) || scaleMin <= 0 || scaleMax <= scaleMin) {
    return [1, 10, 100]
  }
  
  const splits: number[] = []
  const logMin = Math.floor(Math.log10(scaleMin))
  const logMax = Math.ceil(Math.log10(scaleMax))
  
  // Limit to max 6 orders of magnitude to prevent array overflow
  const maxOrders = 6
  const startExp = Math.max(logMin, logMax - maxOrders)
  const endExp = Math.min(logMax, logMin + maxOrders)
  
  // Generate splits at powers of 10
  for (let exp = startExp; exp <= endExp; exp++) {
    const val = Math.pow(10, exp)
    if (val >= scaleMin * 0.9 && val <= scaleMax * 1.1) {
      splits.push(val)
    }
  }
  
  // Ensure we have at least 2 splits
  if (splits.length < 2) {
    splits.length = 0
    splits.push(scaleMin, scaleMax)
  }
  
  return splits
}

// Safe min/max for arrays - avoids stack overflow from spread operator with large arrays
const safeArrayMin = (arr: number[]): number => {
  if (arr.length === 0) return 0
  let min = arr[0]
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i]
  }
  return min
}

const safeArrayMax = (arr: number[]): number => {
  if (arr.length === 0) return 0
  let max = arr[0]
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i]
  }
  return max
}

const computeRange = (
  arrays: ArrayLike<number | null | undefined>[],
  scale: "linear" | "log"
): { min: number; max: number } | null => {
  let min = Infinity
  let max = -Infinity

  for (const arr of arrays) {
    const len = arr?.length ?? 0
    for (let i = 0; i < len; i++) {
      const v = (arr as any)[i]
      if (typeof v !== "number" || !Number.isFinite(v)) continue
      if (scale === "log" && v <= 0) continue
      if (v < min) min = v
      if (v > max) max = v
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || min === Infinity || max === -Infinity) {
    return null
  }

  if (scale === "log") {
    const padding = 0.1
    const paddedMin = Math.max(min / (1 + padding), min * 0.85)
    const paddedMax = max * (1 + padding)
    const safeMin = Math.max(1e-9, paddedMin)
    const safeMax = paddedMax > safeMin ? paddedMax : safeMin * 10
    return { min: safeMin, max: safeMax }
  }

  const span = max - min
  const padding = span > 0 ? span * 0.08 : Math.abs(max || 1) * 0.05
  return { min: min - padding, max: max + padding }
}

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
  const minVal = safeArrayMin(finiteVals)
  const maxVal = safeArrayMax(finiteVals)

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
  onYRangeChange,
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
  const [hoveredSegment, setHoveredSegment] = useState<{ index: number; width: number; x0: number; x1: number; y: number } | null>(null)
  const [hoveredPeak, setHoveredPeak] = useState<{ index: number; time: number; amplitude: number } | null>(null)
  const [hoveredDataPoint, setHoveredDataPoint] = useState<{ x: number; values: { label: string; value: number | null | undefined; color: string }[] } | null>(null)
  // State for custom scatter point hover (DoublePeakScatter)
  const [hoveredScatterPoint, setHoveredScatterPoint] = useState<{ index: number; x: number; y: number; label: string } | null>(null)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  
  // RAF-based throttling for cursor updates
  const rafIdRef = useRef<number | null>(null)
  const lastCursorUpdateRef = useRef<number>(0)

  // Refs for lasso state to avoid re-renders/re-initialization
  const lassoPathRef = useRef<{ x: number; y: number }[]>([])
  const isDrawingLassoRef = useRef(false)

  // Refs for props to access in uPlot hooks without recreating options
  const selectedIndicesRef = useRef(selectedIndices)
  const filteredIndicesRef = useRef(filteredIndices)
  const enableLassoRef = useRef(enableLasso)
  const xScaleTypeRef = useRef(xScaleType)
  
  // Refs for scale-related props to avoid recreating options on scale change
  const yScaleTypeRef = useRef(yScaleType)
  const yRangeRef = useRef(yRange)
  const xRangeRef = useRef(xRange)
  const lastReportedXRangeRef = useRef<{ min: number; max: number } | null>(null)
  const lastReportedYRangeRef = useRef<{ min: number; max: number } | null>(null)
  const rangesEqual = (a: { min: number; max: number } | null | undefined, b: { min: number; max: number } | null | undefined, eps = 1e-6) => {
    if (!a || !b) return false
    return Math.abs(a.min - b.min) < eps && Math.abs(a.max - b.max) < eps
  }

  // Ref to track if we are currently updating scale from props
  // This prevents the setScale hook from firing callbacks when we programmatically set the scale
  const isUpdatingScaleRef = useRef(false)

  // Cache Sets for performance in draw loop
  const filteredSetRef = useRef<Set<number> | null>(null)
  const selectedSetRef = useRef<Set<number> | null>(null)

  const { theme } = useTheme()
  const isDark = theme === "dark"

  // Sync refs with props and update cached Sets
  selectedIndicesRef.current = selectedIndices
  filteredIndicesRef.current = filteredIndices
  enableLassoRef.current = enableLasso
  yScaleTypeRef.current = yScaleType
  xScaleTypeRef.current = xScaleType
  yRangeRef.current = yRange
  xRangeRef.current = xRange

  // Update Sets when arrays change (shallow comparison is enough if references change)
  useMemo(() => {
    filteredSetRef.current = filteredIndices ? new Set(filteredIndices) : null
  }, [filteredIndices])

  useMemo(() => {
    selectedSetRef.current = selectedIndices && selectedIndices.length > 0 ? new Set(selectedIndices) : null
  }, [selectedIndices])

  // Apply y-range updates immediately when provided, then redraw
  // This handles both yRange changes and yScaleType changes imperatively
  useEffect(() => {
    const u = uPlotInstanceRef.current
    if (!u) return
    
    // For scale type changes, we need to update the scale distribution
    // uPlot doesn't support changing distr dynamically, so we need to recreate
    // But we can at least update the range efficiently
    const range = yRangeRef.current
    if (range) {
      let min = range.min
      let max = range.max
      
      // Validate inputs first
      if (!Number.isFinite(min) || !Number.isFinite(max)) return
      
      // For log scale, ensure positive values
      if (yScaleTypeRef.current === "log") {
        if (min <= 0) min = 1e-3
        if (max <= 0) max = 1e-2
        if (max <= min) max = min * 10
      } else {
        // For linear scale, ensure max > min
        if (max <= min) {
          const padding = Math.abs(min) * 0.1 || 1
          max = min + padding
        }
      }
      
      if (Number.isFinite(min) && Number.isFinite(max) && max > min) {
        isUpdatingScaleRef.current = true
        u.setScale("y", { min, max })
        isUpdatingScaleRef.current = false
        
        // Batch the redraw with RAF to avoid multiple redraws
        requestAnimationFrame(() => {
          if (uPlotInstanceRef.current) {
            uPlotInstanceRef.current.redraw()
          }
        })
      }
    }
  }, [yRange, yScaleType])

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
    
    // Cleanup on unmount - CRITICAL for preventing memory leaks
    return () => {
      // Cancel any pending animation frames
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current)
        rafIdRef.current = null
      }
      
      // Destroy the uPlot instance to free memory
      cleanupUPlotInstance(uPlotInstanceRef)
      
      // Clear all state refs
      lassoPathRef.current = []
      isDrawingLassoRef.current = false
    }
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

  // Extract series config (ignoring data) to prevent unnecessary re-initializations
  const seriesConfig = useMemo(() => {
    return series.map(s => ({
      label: s.label,
      color: s.color,
      width: s.width,
      dash: s.dash,
      points: s.points,
      pointSize: s.pointSize,
      bar: s.bar,
      barWidth: s.barWidth,
      dataIndices: s.dataIndices
    }))
  }, [series])

  // Use ref for widthSegments to access in draw hook without triggering options recreation
  const widthSegmentsRef = useRef(widthSegments)
  useEffect(() => {
    widthSegmentsRef.current = widthSegments
  }, [widthSegments])

  const opts: uPlot.Options = useMemo(() => {
    const gridColor = isDark ? "rgba(248,250,252,0.10)" : "rgba(15,23,42,0.06)"
    const axisColor = isDark ? "rgba(248,250,252,0.88)" : "rgba(15,23,42,0.65)"
    const axisFont = "600 11px 'Inter', 'Segoe UI', system-ui, sans-serif"
    const axisLabelFont = "600 12px 'Inter', 'Segoe UI', system-ui, sans-serif"

    // Dynamic x-axis label unit based on current visible span (xData is in minutes)
    // Use xRange if available, otherwise fallback to basic defaults to avoid xData dependency
    const currentXMin = xRange?.min ?? 0
    const currentXMax = xRange?.max ?? 10 // Arbitrary default if no range/data, will be updated by scale
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

            // Ensure valid positive values for log scale
            if (!Number.isFinite(min) || min <= 0) {
              min = Number.isFinite(dataMin) && dataMin > 0 ? dataMin : 1e-3
            }
            if (!Number.isFinite(max) || max <= 0) {
              max = Number.isFinite(dataMax) && dataMax > 0 ? dataMax : 1
            }
            // Ensure max > min with a minimum span
            if (max <= min) {
              max = min * 10
            }
            // Final safety check
            if (!Number.isFinite(min) || !Number.isFinite(max) || min <= 0 || max <= min) {
              return [1e-3, 1]
            }
            
            // CRITICAL: Limit the ratio to prevent uPlot's logAxisSplits from creating too many elements
            // A ratio > 10000 (4 orders of magnitude) can cause RangeError: Invalid array length
            const MAX_LOG_RATIO = 10000
            const ratio = max / min
            if (ratio > MAX_LOG_RATIO) {
              const geomMean = Math.sqrt(min * max)
              const halfOrders = Math.log10(MAX_LOG_RATIO) / 2
              min = geomMean / Math.pow(10, halfOrders)
              max = geomMean * Math.pow(10, halfOrders)
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
          // Validate xRange before applying - invalid ranges cause uPlot axis split errors
          ...((xRange && 
               Number.isFinite(xRange.min) && 
               Number.isFinite(xRange.max) && 
               xRange.max > xRange.min) 
            ? { min: xRange.min, max: xRange.max } 
            : {}),
        },
        y: {
          // Validate yRange before applying
          ...((yRange && 
               Number.isFinite(yRange.min) && 
               Number.isFinite(yRange.max) && 
               yRange.max > yRange.min &&
               (yScaleType !== "log" || (yRange.min > 0 && yRange.max > 0)))
            ? { auto: false, min: yRange.min, max: yRange.max } 
            : { auto: true }),
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
          // Use safe splits function for log scale to prevent RangeError
          ...(xScaleType === "log" ? { splits: safeLogAxisSplits } : {}),
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
          // Use safe splits function for log scale to prevent RangeError
          ...(yScaleType === "log" ? { splits: safeLogAxisSplits } : {}),
        },
      ],
      legend: {
        show: !!legend,
        stroke: isDark ? "rgba(248,250,252,0.95)" : "#1f2937",
        fill: isDark ? "rgba(0,0,0,0)" : "rgba(255,255,255,0)",
      },
      series: [
        {}, // x axis
        ...seriesConfig.map(s => {
          // Determine if this is a point-only series (no line)
          const isPointOnly = s.points && (s.width === 0 || customScatter)
          const lineWidth = s.bar ? 0 : (customScatter && s.points ? 0 : (s.width ?? 1.5))
          
          return {
            label: s.label,
            stroke: isPointOnly ? "transparent" : s.color, // No stroke for point-only series
            width: lineWidth,
            dash: s.dash,
            // Only apply gradient fill for line series, not scatter/point-only series
            fill: isPointOnly ? undefined : (self: uPlot, seriesIdx: number) => {
              const ctx = self.ctx
              const { top, height } = self.bbox
              if (!Number.isFinite(top) || !Number.isFinite(height)) {
                return null
              }
              const gradient = ctx.createLinearGradient(0, top, 0, top + height)
              gradient.addColorStop(0, isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)")
              gradient.addColorStop(1, "rgba(0,0,0,0)")
              return gradient
            },
            points: s.points ? {
              // Hide default points only when we render custom ones ourselves
              show: customScatter ? false : true,
              size: s.pointSize ?? 5,
              stroke: s.color,
              fill: s.color,
            } : undefined,
          }
        }),
      ],
      hooks: {
        setScale: [
          (u: uPlot, key: string) => {
            // Don't trigger callback if we initiated the update from props
            if (isUpdatingScaleRef.current) return

            if (key === "x" && onXRangeChange) {
              const min = u.scales.x.min!
              const max = u.scales.x.max!
              if (Number.isFinite(min) && Number.isFinite(max)) {
                const next = { min, max }
                if (rangesEqual(lastReportedXRangeRef.current, next) || rangesEqual(xRangeRef.current || null, next)) {
                  lastReportedXRangeRef.current = next
                  return
                }
                lastReportedXRangeRef.current = next
                onXRangeChange(next)
              }
            } else if (key === "y" && onYRangeChange) {
              const min = u.scales.y.min!
              const max = u.scales.y.max!
              if (Number.isFinite(min) && Number.isFinite(max)) {
                const next = { min, max }
                if (rangesEqual(lastReportedYRangeRef.current, next) || rangesEqual(yRangeRef.current || null, next)) {
                  lastReportedYRangeRef.current = next
                  return
                }
                lastReportedYRangeRef.current = next
                onYRangeChange(next)
              }
            }
          }
        ],
        setCursor: [
          (u: uPlot) => {
            // RAF-based throttling to prevent excessive updates with large datasets
            const now = performance.now()
            if (now - lastCursorUpdateRef.current < 16) { // ~60fps max
              if (!rafIdRef.current) {
                rafIdRef.current = requestAnimationFrame(() => {
                  rafIdRef.current = null
                  // Re-trigger cursor processing on next frame
                  if (uPlotInstanceRef.current) {
                    const idx = uPlotInstanceRef.current.cursor.idx
                    processCursorUpdate(uPlotInstanceRef.current, idx)
                  }
                })
              }
              return
            }
            lastCursorUpdateRef.current = now
            
            const idx = u.cursor.idx
            processCursorUpdate(u, idx)
            
            function processCursorUpdate(uPlotInst: uPlot, cursorIdx: number | null | undefined) {
              if (cursorIdx != null && cursorIdx >= 0) {
                const xVal = uPlotInst.data[0][cursorIdx]
                const values = seriesConfig.map((s, i) => ({
                  label: s.label,
                  value: uPlotInst.data[i + 1][cursorIdx],
                  color: s.color
                })).filter(v => v.value != null && Number.isFinite(v.value))

                // For customScatter (scatter plot), find the point series and show its data
                const pointSeriesIdx = seriesConfig.findIndex(s => s.points)
                if (pointSeriesIdx >= 0) {
                  const pointYVal = uPlotInst.data[pointSeriesIdx + 1][cursorIdx]
                  const pointLabel = seriesConfig[pointSeriesIdx].label
                  
                  if (pointYVal != null && Number.isFinite(pointYVal as number)) {
                    // For customScatter (DoublePeakScatter), use hoveredScatterPoint
                    if (customScatter) {
                      setHoveredScatterPoint({
                        index: cursorIdx,
                        x: xVal as number,
                        y: pointYVal as number,
                        label: pointLabel
                      })
                      setHoveredPeak(null)
                    } else {
                      // For standard peak overlay, use hoveredPeak
                      setHoveredPeak({
                        index: cursorIdx,
                        time: xVal as number,
                        amplitude: pointYVal as number
                      })
                      setHoveredScatterPoint(null)
                    }
                  } else {
                    setHoveredPeak(null)
                    setHoveredScatterPoint(null)
                  }
                } else {
                  setHoveredPeak(null)
                  setHoveredScatterPoint(null)
                }

                if (values.length > 0) {
                  setHoveredDataPoint({ x: xVal, values })
                } else {
                  setHoveredDataPoint(null)
                }
              } else {
                setHoveredDataPoint(null)
                setHoveredPeak(null)
                setHoveredScatterPoint(null)
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
                ; (legendEl as HTMLElement).style.color = legendTextColor
              const legendItems = legendEl.querySelectorAll('.u-legend-item')
              legendItems.forEach((item) => {
                ; (item as HTMLElement).style.color = legendTextColor
              })
            }

            // Wheel zoom (disabled when lasso is active)
            over.addEventListener("wheel", (e: WheelEvent) => {
              if (enableLassoRef.current) return

              e.preventDefault()
              const rect = over.getBoundingClientRect()
              const leftPx = e.clientX - rect.left
              const topPx = e.clientY - rect.top

              if (enableYAxisZoom && e.shiftKey) {
                const yVal = u.posToVal(topPx, "y")
                const factor = e.deltaY < 0 ? 0.85 : 1 / 0.85
                const min = u.scales.y.min!
                const max = u.scales.y.max!
                const nyMin = yVal - (yVal - min) * factor
                const nyMax = yVal + (max - yVal) * factor
                u.setScale("y", { min: nyMin, max: nyMax })
              } else if (enableYAxisZoom) {
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
                  const yVal = u.posToVal(topPx, "y")
                  const factor = e.deltaY < 0 ? 0.85 : 1 / 0.85
                  const min = u.scales.y.min!
                  const max = u.scales.y.max!
                  const nyMin = yVal - (yVal - min) * factor
                  const nyMax = yVal + (max - yVal) * factor
                  u.setScale("y", { min: nyMin, max: nyMax })
                } else {
                  const xVal = u.posToVal(leftPx, "x")
                  const factor = e.deltaY < 0 ? 0.85 : 1 / 0.85
                  const min = u.scales.x.min!
                  const max = u.scales.x.max!
                  const nxMin = xVal - (xVal - min) * factor
                  const nxMax = xVal + (max - xVal) * factor
                  u.setScale("x", { min: nxMin, max: nxMax })
                }
              } else {
                const xVal = u.posToVal(leftPx, "x")
                const factor = e.deltaY < 0 ? 0.85 : 1 / 0.85
                const min = u.scales.x.min!
                const max = u.scales.x.max!
                const nxMin = xVal - (xVal - min) * factor
                const nxMax = xVal + (max - xVal) * factor
                u.setScale("x", { min: nxMin, max: nxMax })
              }
            }, { passive: false })

            // Double-click to reset zoom
            over.addEventListener("dblclick", (e: MouseEvent) => {
              if (enableLassoRef.current) return

              e.preventDefault()
              e.stopPropagation()

              if (onResetZoom) {
                onResetZoom()
              } else {
                const xSeries = u.data[0] as number[]
                const xRangeReset = computeRange([xSeries], xScaleTypeRef.current)
                if (xRangeReset) {
                  u.setScale("x", xRangeReset)
                  onXRangeChange?.(xRangeReset)
                }

                const yRangeReset = computeRange(
                  u.data.slice(1) as ArrayLike<number | null | undefined>[],
                  yScaleTypeRef.current
                )
                if (yRangeReset) {
                  u.setScale("y", yRangeReset)
                  onYRangeChange?.(yRangeReset)
                } else {
                  const fallback = yScaleTypeRef.current === "log"
                    ? { min: 1e-3, max: 1 }
                    : { min: 0, max: 100 }
                  u.setScale("y", fallback)
                  onYRangeChange?.(fallback)
                }
              }
            })

            // Mouse move for width segment hover detection
            over.addEventListener("mousemove", (e: MouseEvent) => {
              if (enableLassoRef.current) {
                setHoveredSegment(null)
                return
              }

              const segments = widthSegmentsRef.current
              if (!segments || segments.length === 0) {
                setHoveredSegment(null)
                return
              }

              const overRect = over.getBoundingClientRect()
              const containerRect = containerRef.current?.getBoundingClientRect()
              const mouseX = e.clientX - overRect.left
              const mouseY = e.clientY - overRect.top

              const containerX = containerRect ? e.clientX - containerRect.left : mouseX
              const containerY = containerRect ? e.clientY - containerRect.top : mouseY

              setMousePos({ x: containerX, y: containerY })

              let found = false
              for (let i = 0; i < segments.length; i += 1) {
                const seg = segments[i]
                const x0px = u.valToPos(seg.x0, "x", true)
                const x1px = u.valToPos(seg.x1, "x", true)
                const ypx = u.valToPos(seg.y, "y", true)

                const hoverThresholdY = 8
                const hoverPaddingX = 5
                if (
                  mouseX >= x0px - hoverPaddingX &&
                  mouseX <= x1px + hoverPaddingX &&
                  Math.abs(mouseY - ypx) <= hoverThresholdY
                ) {
                  const widthValue = seg.width ?? ((seg.x1 - seg.x0) * 60 * 1000) // Convert to ms if not provided
                  setHoveredSegment({ 
                    index: i, 
                    width: widthValue,
                    x0: seg.x0,
                    x1: seg.x1,
                    y: seg.y
                  })
                  found = true
                  break
                }
              }

              if (!found) {
                setHoveredSegment(null)
              }
            })

            over.addEventListener("mouseleave", () => {
              setHoveredSegment(null)
              setMousePos(null)
              setHoveredDataPoint(null)
              setHoveredPeak(null)
              setHoveredScatterPoint(null)
            })

            over.style.cursor = "crosshair"
          }
        ],
        draw: [
          (u: uPlot) => {
            const ctx = u.ctx

            // Update legend text color
            const legendEl = u.root.querySelector('.u-legend')
            if (legendEl) {
              const legendTextColor = isDark ? "rgba(248,250,252,0.95)" : "#1f2937"
                ; (legendEl as HTMLElement).style.color = legendTextColor
              const legendItems = legendEl.querySelectorAll('.u-legend-item')
              legendItems.forEach((item) => {
                ; (item as HTMLElement).style.color = legendTextColor
              })
            }

            // Draw bar-style series
            if (seriesConfig && seriesConfig.length > 0) {
              const xVals = (u.data[0] as number[]) || []
              seriesConfig.forEach((s, sIdx) => {
                if (!s.bar) return
                const yVals = (u.data[sIdx + 1] as (number | null | undefined)[]) || []
                if (!xVals.length || !yVals.length) return

                ctx.save()
                ctx.fillStyle = s.color
                ctx.strokeStyle = s.color
                ctx.lineWidth = 1

                const baseY = yScaleType === "log" ? Math.max(u.scales.y.min ?? 1e-9, 1e-9) : (u.scales.y.min ?? 0)
                const baseYPx = u.valToPos(baseY, "y", true)

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

            // Draw width segments
            const segments = widthSegmentsRef.current
            if (segments && segments.length > 0) {
              ctx.save()
              const capHeight = 8

              segments.forEach((seg, idx) => {
                const x0px = Math.round(u.valToPos(seg.x0, "x", true))
                const x1px = Math.round(u.valToPos(seg.x1, "x", true))
                const ypx = Math.round(u.valToPos(seg.y, "y", true))

                const isHovered = hoveredSegment?.index === idx

                ctx.lineWidth = isHovered ? 3 : 2
                // Width segment colors - using accent orange
                const baseColor = isDark ? "#fb923c" : "#f97316"
                const hoverColor = isDark ? "#fdba74" : "#ea580c"

                ctx.strokeStyle = isHovered ? hoverColor : baseColor

                ctx.beginPath()
                ctx.moveTo(x0px, ypx)
                ctx.lineTo(x1px, ypx)
                ctx.stroke()

                ctx.beginPath()
                ctx.moveTo(x0px, ypx - capHeight / 2)
                ctx.lineTo(x0px, ypx + capHeight / 2)
                ctx.stroke()

                ctx.beginPath()
                ctx.moveTo(x1px, ypx - capHeight / 2)
                ctx.lineTo(x1px, ypx + capHeight / 2)
                ctx.stroke()
              })
              ctx.restore()
            }

            // Draw horizontal threshold lines
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

            // Draw vertical threshold lines
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

            // Draw offset annotation
            if (offsetInfoRef.current) {
              const { offset, unit, decimals } = offsetInfoRef.current
              ctx.save()

              let offsetText = ""
              if (unit === "ms") {
                const msDecimals = decimals ?? 0
                offsetText = `+ ${offset.toFixed(msDecimals)} ms`
              } else if (unit === "s") {
                offsetText = `+ ${offset.toFixed(1)} s`
              } else {
                offsetText = `+ ${offset.toFixed(2)} min`
              }

              ctx.font = "600 11px 'Inter', 'Segoe UI', system-ui, sans-serif"
              ctx.fillStyle = isDark ? "rgba(248,250,252,0.75)" : "rgba(15,23,42,0.60)"
              ctx.textAlign = "left"
              ctx.textBaseline = "bottom"

              const plotLeft = u.bbox.left
              const plotBottom = u.bbox.top + u.bbox.height
              const xPos = plotLeft + 5
              const yPos = plotBottom - 3

              ctx.fillText(offsetText, xPos, yPos)
              ctx.restore()
            }

            // Draw lasso path
            const lassoPath = lassoPathRef.current
            const isDrawingLasso = isDrawingLassoRef.current

            if (lassoPath.length > 1) {
              ctx.save()
              ctx.strokeStyle = isDark ? "#fb923c" : "#f97316"  // accent orange
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
                ctx.closePath()
                ctx.stroke()
              }
              ctx.restore()
            }

            // Draw glowing peaks for standard series (not customScatter)
            if (!customScatter) {
              // Get visible index range from uPlot to optimize rendering
              // Safety: ensure indices are within bounds and limit max points to prevent OOM
              const dataLen = u.data[0] ? u.data[0].length : 0
              if (dataLen === 0) return
              
              let iMin = (u as any).idx ? (u as any).idx[0] : 0
              let iMax = (u as any).idx ? (u as any).idx[1] : dataLen - 1
              
              // Clamp to valid range
              iMin = Math.max(0, Math.min(iMin, dataLen - 1))
              iMax = Math.max(iMin, Math.min(iMax, dataLen - 1))
              
              // Safety limit: don't render more than 50k points to prevent OOM
              const maxPointsToRender = 50000
              if (iMax - iMin > maxPointsToRender) {
                // Downsample by skipping points
                const step = Math.ceil((iMax - iMin) / maxPointsToRender)
                
                seriesConfig.forEach((s, sIdx) => {
                  if (!s.points) return
                  const xVals = u.data[0] as number[]
                  const yVals = u.data[sIdx + 1] as (number | null | undefined)[]
                  if (!xVals || !yVals) return

                  ctx.save()

                  const xMinPx = u.bbox.left - 10
                  const xMaxPx = u.bbox.left + u.bbox.width + 10
                  const yMinPx = u.bbox.top - 10
                  const yMaxPx = u.bbox.top + u.bbox.height + 10

                  // Single pass for downsampled data (no glow to save memory)
                  ctx.beginPath()
                  const size = s.pointSize ?? 3
                  
                  for (let i = iMin; i <= iMax; i += step) {
                    const xVal = xVals[i]
                    const yVal = yVals[i]
                    if (yVal == null || !Number.isFinite(yVal) || !Number.isFinite(xVal)) continue

                    const xPx = u.valToPos(xVal, "x", true)
                    const yPx = u.valToPos(yVal, "y", true)

                    // Strict bounds check including NaN/Infinity
                    if (!Number.isFinite(xPx) || !Number.isFinite(yPx)) continue
                    if (xPx < xMinPx || xPx > xMaxPx || yPx < yMinPx || yPx > yMaxPx) continue

                    ctx.moveTo(xPx + size, yPx)
                    ctx.arc(xPx, yPx, size, 0, 2 * Math.PI)
                  }
                  
                  ctx.fillStyle = s.color
                  ctx.fill()
                  ctx.restore()
                })
              } else {
                // Normal rendering for reasonable point counts
                seriesConfig.forEach((s, sIdx) => {
                  if (!s.points) return
                  const xVals = u.data[0] as number[]
                  const yVals = u.data[sIdx + 1] as (number | null | undefined)[]
                  if (!xVals || !yVals) return

                  ctx.save()

                  const xMinPx = u.bbox.left - 10
                  const xMaxPx = u.bbox.left + u.bbox.width + 10
                  const yMinPx = u.bbox.top - 10
                  const yMaxPx = u.bbox.top + u.bbox.height + 10

                  const drawPass = (isGlow: boolean) => {
                    const size = isGlow ? 8 : (s.pointSize ?? 4)

                    const renderPoint = (i: number) => {
                      const xVal = xVals[i]
                      const yVal = yVals[i]
                      if (yVal == null || !Number.isFinite(yVal) || !Number.isFinite(xVal)) return

                      const xPx = u.valToPos(xVal, "x", true)
                      const yPx = u.valToPos(yVal, "y", true)

                      // Strict bounds check including NaN/Infinity
                      if (!Number.isFinite(xPx) || !Number.isFinite(yPx)) return
                      if (xPx < xMinPx || xPx > xMaxPx || yPx < yMinPx || yPx > yMaxPx) return

                      ctx.moveTo(xPx + size, yPx)
                      ctx.arc(xPx, yPx, size, 0, 2 * Math.PI)
                    }

                    if (s.dataIndices) {
                      // Optimization: Iterate only provided indices
                      // Binary search to find start index in dataIndices that is >= iMin
                      let left = 0
                      let right = s.dataIndices.length - 1
                      let startIdx = -1

                      while (left <= right) {
                        const mid = (left + right) >> 1
                        if (s.dataIndices[mid] >= iMin) {
                          startIdx = mid
                          right = mid - 1
                        } else {
                          left = mid + 1
                        }
                      }

                      if (startIdx !== -1) {
                        for (let k = startIdx; k < s.dataIndices.length; k++) {
                          const i = s.dataIndices[k]
                          if (i > iMax) break
                          renderPoint(i)
                        }
                      }
                    } else {
                      // Default: Iterate visible range
                      for (let i = iMin; i <= iMax; i++) {
                        renderPoint(i)
                      }
                    }
                  }

                  // Draw Glow (Halo)
                  ctx.beginPath()
                  drawPass(true)
                  ctx.fillStyle = s.color
                  ctx.globalAlpha = 0.2
                  ctx.fill()

                  // Draw inner point (solid)
                  ctx.beginPath()
                  drawPass(false)
                  ctx.globalAlpha = 1.0
                  ctx.fillStyle = s.color
                  ctx.fill()

                  ctx.restore()
                })
              }
            }

            // Custom point rendering with three states (Double Peak only)
            const currentXData = u.data[0] as number[]
            const firstSeries = seriesConfig.find(s => s.points)
            if (customScatter && firstSeries && currentXData && currentXData.length > 0) {
              ctx.save()
              ctx.beginPath();
              ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
              ctx.clip();

              // Get visible index range from uPlot to optimize rendering
              // Safety: clamp to valid range
              const dataLen = currentXData.length
              let iMin = (u as any).idx ? (u as any).idx[0] : 0
              let iMax = (u as any).idx ? (u as any).idx[1] : dataLen - 1
              iMin = Math.max(0, Math.min(iMin, dataLen - 1))
              iMax = Math.max(iMin, Math.min(iMax, dataLen - 1))

              const filteredSet = filteredSetRef.current
              const selectedSet = selectedSetRef.current

              const hasFilters = filteredSet && filteredSet.size < xData.length

              const selectedPath = new Path2D();
              const normalPath = new Path2D();
              const filteredPath = new Path2D();

              let hasSelected = false;
              let hasNormal = false;
              let hasFiltered = false;

              const xMin = u.bbox.left - 10;
              const xMax = u.bbox.left + u.bbox.width + 10;
              const yMin = u.bbox.top - 10;
              const yMax = u.bbox.top + u.bbox.height + 10;

              const seriesIdx = seriesConfig.findIndex(s => s.points)
              if (seriesIdx === -1) {
                ctx.restore()
                return
              }
              const yValues = u.data[seriesIdx + 1] as number[]

              // Safety limit for custom scatter too
              const maxPoints = 50000
              const step = (iMax - iMin > maxPoints) ? Math.ceil((iMax - iMin) / maxPoints) : 1

              for (let i = iMin; i <= iMax; i += step) {
                const xVal = currentXData[i]
                const yVal = yValues[i]
                if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue

                const xPx = u.valToPos(xVal, "x", true)
                const yPx = u.valToPos(yVal as number, "y", true)

                // Strict bounds check including NaN/Infinity
                if (!Number.isFinite(xPx) || !Number.isFinite(yPx)) continue
                if (xPx < xMin || xPx > xMax || yPx < yMin || yPx > yMax) continue;

                const isSelected = selectedSet?.has(i)
                const passesFilter = !hasFilters || filteredSet?.has(i)

                if (isSelected) {
                  selectedPath.moveTo(xPx + 5, yPx);
                  selectedPath.arc(xPx, yPx, 5, 0, 2 * Math.PI);
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
                ctx.fillStyle = isDark ? "rgba(251, 146, 60, 0.8)" : "rgba(249, 115, 22, 0.8)"  // accent orange
                ctx.strokeStyle = isDark ? "#fb923c" : "#f97316"
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
  }, [width, height, isDark, seriesConfig, xLabel, yLabel, onResetZoom, hLines, vLines, xRange, onXRangeChange, onYRangeChange, xScaleType, yScaleType, legend, enableYAxisZoom, customScatter, hoveredSegment?.index, xData.length, yRange])
  // Note: yRange is intentionally excluded - we handle it via setScale() in useEffect
  // yScaleType must be included as it changes the scale distribution (distr: 3 for log)

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
        isUpdatingScaleRef.current = true
        u.setScale("x", { min: xRange.min, max: xRange.max })
        isUpdatingScaleRef.current = false
      }
    }
  }, [data, xRange]) // Update when data or xRange changes

  // Keep last reported ranges in sync when props change externally
  useEffect(() => {
    if (xRange && Number.isFinite(xRange.min) && Number.isFinite(xRange.max)) {
      lastReportedXRangeRef.current = { ...xRange }
    }
  }, [xRange])

  useEffect(() => {
    if (yRange && Number.isFinite(yRange.min) && Number.isFinite(yRange.max)) {
      lastReportedYRangeRef.current = { ...yRange }
    }
  }, [yRange])

  const handleExportImage = () => {
    const u = uPlotInstanceRef.current
    if (!u) return

    const canvas = u.ctx.canvas
    const link = document.createElement("a")
    link.download = `chart-${Date.now()}.png`
    link.href = canvas.toDataURL("image/png")
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    toast.success("Chart exported successfully", {
      description: "The image has been saved to your downloads folder.",
    })
  }

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

  // Use a stable key that only changes when scale type changes
  // This helps React properly manage chart recreation
  const chartKey = `${xScaleType}-${yScaleType}`

  return (
    <div ref={containerRef} className={`${className} group relative`} style={{ minHeight: 0 }}>
      <UplotReact key={chartKey} options={opts} data={data} />

      <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
        <Button
          variant="secondary"
          size="icon"
          className="h-8 w-8 bg-background/80 backdrop-blur-sm shadow-sm border"
          onClick={handleExportImage}
          title="Save Image"
        >
          <Camera className="h-4 w-4" />
        </Button>
      </div>

      {/* Unified hover info bar - positioned above x-axis area */}
      {(hoveredSegment || hoveredPeak || hoveredScatterPoint) && (
        <div
          className="absolute left-0 right-0 z-40 px-3 py-1.5 border-t border-b flex items-center gap-4 text-xs"
          style={{
            bottom: 50, // Position above x-axis label area
            backgroundColor: isDark ? "rgba(15, 23, 42, 0.95)" : "rgba(255, 255, 255, 0.97)",
            borderColor: isDark ? "rgba(71, 85, 105, 0.4)" : "rgba(203, 213, 225, 0.6)",
            backdropFilter: "blur(8px)",
          }}
        >
          {/* Peak Width Info */}
          {hoveredSegment && (
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full shadow-sm" style={{ backgroundColor: isDark ? "#fb923c" : "#f97316" }}></div>
              <span className="text-muted-foreground">Peak Width:</span>
              <span className="font-semibold" style={{ color: isDark ? "#fdba74" : "#ea580c" }}>
                {hoveredSegment.width.toFixed(3)} ms
              </span>
              <span className="text-muted-foreground/60 ml-1">
                ({(hoveredSegment.x0 * 60).toFixed(2)}s – {(hoveredSegment.x1 * 60).toFixed(2)}s)
              </span>
            </div>
          )}
          
          {/* Scatter Point Info (for DoublePeakScatter) */}
          {hoveredScatterPoint && !hoveredSegment && (
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full shadow-sm" style={{ backgroundColor: isDark ? "#94a3b8" : "#64748b" }}></div>
              <span className="text-muted-foreground">{hoveredScatterPoint.label}:</span>
              <span className="font-semibold">
                {hoveredScatterPoint.y.toFixed(2)} ms
              </span>
              <span className="text-muted-foreground/80">at</span>
              <span className="font-medium">
                {hoveredScatterPoint.x >= 1 
                  ? `${hoveredScatterPoint.x.toFixed(3)} min`
                  : `${(hoveredScatterPoint.x * 60).toFixed(2)} s`
                }
              </span>
            </div>
          )}
          
          {/* Peak Point Info (for preprocessing peaks) */}
          {hoveredPeak && !hoveredSegment && !hoveredScatterPoint && (
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full shadow-sm" style={{ backgroundColor: isDark ? "#f87171" : "#ef4444" }}></div>
              <span className="text-muted-foreground">Peak:</span>
              <span className="font-semibold" style={{ color: isDark ? "#f87171" : "#ef4444" }}>
                {hoveredPeak.amplitude.toFixed(2)}
              </span>
              <span className="text-muted-foreground/80">at</span>
              <span className="font-medium">
                {hoveredPeak.time >= 1 
                  ? `${hoveredPeak.time.toFixed(3)} min`
                  : `${(hoveredPeak.time * 60).toFixed(2)} s`
                }
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
