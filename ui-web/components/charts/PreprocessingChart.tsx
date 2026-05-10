"use client"

import { useEffect, useMemo, useState, useRef, useCallback } from "react"
import { Loader2 } from "lucide-react"
import { apiClient } from "@/lib/apiClient"
import { UPlotChart } from "./UPlotChart"
import { UPlotHistogram } from "./UPlotHistogram"
import { HistogramCard } from "./HistogramCard"
import { useTheme } from "@/hooks/use-theme"
import { parseResultBinary } from "@/lib/binaryParsers"

interface PreprocessingChartProps {
  resultId: string
  filteredResultId?: string | null
  className?: string
  timeResolution?: number
  peakTimes?: number[]
  peakAmplitudes?: number[]
  peakIntervals?: number[]  // Intervals between peaks in milliseconds
  peakProperties?: {
    prominences?: number[]
    left_ips: number[]
    right_ips: number[]
    widths: number[]
    width_heights?: number[]
  } | null
  prominenceThreshold?: number
  distance?: number
  widthMs?: string
  initialHistograms?: {
    amplitude: { bins: number[]; counts: number[]; bin_edges?: number[] }
    width: { bins: number[]; counts: number[]; bin_edges?: number[] }
    interval: { bins: number[]; counts: number[]; bin_edges?: number[] }
  } | null
}

type NumericArray = number[] | Float32Array | Float64Array
type WorkerRange = { min: number; max: number } | null

type WorkerReadyMessage = {
  type: 'READY'
  payload: {
    totalPoints: number
    timeRange: WorkerRange
  }
}

type WorkerRangeMessage = {
  type: 'RANGE_DATA'
  requestId?: string
  payload: {
    time: Float64Array
    raw: Float32Array
    filtered: Float32Array | null
    displayedPoints: number
    totalPoints: number
    visibleRange: WorkerRange
    decimated: boolean
    widthSegments: {
      x0: Float64Array
      x1: Float64Array
      y: Float32Array
    } | null
  }
}

type WorkerErrorMessage = {
  type: 'ERROR'
  requestId?: string
  payload: {
    message: string
  }
}

type WorkerMessage = WorkerReadyMessage | WorkerRangeMessage | WorkerErrorMessage

type HistogramMetricKey = "amplitude" | "width" | "interval"
type AxisScaleOption = "linear" | "log"
type HistogramAxisConfig = { xScale: AxisScaleOption; yScale: "linear" | "log" }

const defaultAxisConfig: HistogramAxisConfig = { xScale: "log", yScale: "log" }
const TARGET_POINTS = 2500
const RANGE_OVERSCAN_MULTIPLIER = 3

const cleanNumbers = (values: number[] = []) => values.filter((v) => Number.isFinite(v))
const lowerBound = (values: ArrayLike<number>, target: number) => {
  let lo = 0
  let hi = values.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (values[mid] < target) lo = mid + 1
    else hi = mid
  }
  return lo
}
const upperBound = (values: ArrayLike<number>, target: number) => {
  let lo = 0
  let hi = values.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (values[mid] <= target) lo = mid + 1
    else hi = mid
  }
  return lo
}
const expandRange = (range: WorkerRange, bounds: WorkerRange, multiplier = RANGE_OVERSCAN_MULTIPLIER): WorkerRange => {
  if (!range || !bounds) return range
  if (
    !Number.isFinite(range.min) ||
    !Number.isFinite(range.max) ||
    !Number.isFinite(bounds.min) ||
    !Number.isFinite(bounds.max) ||
    range.max <= range.min ||
    bounds.max <= bounds.min
  ) {
    return range
  }

  const span = range.max - range.min
  const padding = span * Math.max(0, multiplier - 1) / 2
  return {
    min: Math.max(bounds.min, range.min - padding),
    max: Math.min(bounds.max, range.max + padding),
  }
}
const mean = (values: number[]) => {
  const arr = cleanNumbers(values)
  if (!arr.length) return null
  return arr.reduce((sum, v) => sum + v, 0) / arr.length
}
const std = (values: number[]) => {
  const arr = cleanNumbers(values)
  if (!arr.length) return null
  const m = mean(arr)
  if (m == null) return null
  const variance = arr.reduce((sum, v) => sum + Math.pow(v - m, 2), 0) / arr.length
  return Math.sqrt(variance)
}

export function PreprocessingChart({
  resultId,
  filteredResultId,
  className = "",
  timeResolution,
  peakTimes = [],
  peakAmplitudes = [],
  peakIntervals = [],
  peakProperties = null,
  prominenceThreshold,
  distance,
  widthMs,
  initialHistograms,
}: PreprocessingChartProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<any>(null)
  const [histogramData, setHistogramData] = useState<{
    amplitude: { bins: number[]; counts: number[]; bin_edges?: number[] }
    width: { bins: number[]; counts: number[]; bin_edges?: number[] }
    interval: { bins: number[]; counts: number[]; bin_edges?: number[] }
  } | null>(initialHistograms || null)
  const [histogramLoading, setHistogramLoading] = useState(false)
  const [histogramScales, setHistogramScales] = useState<Record<HistogramMetricKey, HistogramAxisConfig>>({
    amplitude: { ...defaultAxisConfig },
    width: { ...defaultAxisConfig },
    interval: { ...defaultAxisConfig },
  })
  const [histogramConfig, setHistogramConfig] = useState<{
    binCount: number
  }>({
    binCount: 60,
  })
  const updateXAxisScale = useCallback((metric: HistogramMetricKey, value: AxisScaleOption) => {
    setHistogramScales((prev) => ({
      ...prev,
      [metric]: {
        ...prev[metric],
        xScale: value,
      },
    }))
  }, [])
  const updateYAxisScale = useCallback((metric: HistogramMetricKey, value: "linear" | "log") => {
    setHistogramScales((prev) => ({
      ...prev,
      [metric]: {
        ...prev[metric],
        yScale: value,
      },
    }))
  }, [])
  const updateBinCount = useCallback((value: number) => {
    setHistogramConfig((prev) => ({
      ...prev,
      binCount: value,
    }))
  }, [])

  // Displayed data (fed by worker)
  const [xData, setXData] = useState<NumericArray>([])
  const [yOriginal, setYOriginal] = useState<NumericArray>([])
  const [yFiltered, setYFiltered] = useState<NumericArray | null>(null)

  const [widthSegmentsState, setWidthSegmentsState] = useState<{
    x0: Float64Array
    x1: Float64Array
    y: Float32Array
  } | null>(null)

  const [zoomRange, setZoomRange] = useState<WorkerRange>(null)
  const [dynamicDownsampling, setDynamicDownsampling] = useState<boolean>(true)
  const [workerReady, setWorkerReady] = useState(false)

  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)
  const zoomStateFrameRef = useRef<number | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const lastRangeRef = useRef<WorkerRange>(null)
  const pendingRangeRef = useRef<WorkerRange>(null)
  const zoomRangeRef = useRef<WorkerRange>(null)
  const timeRangeRef = useRef<WorkerRange>(null)
  const latestRequestIdRef = useRef<string | null>(null)
  const requestCounterRef = useRef(0)
  const { theme } = useTheme()
  const isDark = theme === "dark"
  const accentColor = isDark ? "#fb923c" : "#f97316"
  // Histogram colors: distinct, visually differentiated palette
  const histogramColors = {
    amplitude: isDark ? "#f87171" : "#ef4444",  // Red/coral for prominence (matches peak dots)
    width: isDark ? "#2dd4bf" : "#14b8a6",      // Teal/green for peak width
    interval: isDark ? "#a78bfa" : "#8b5cf6",   // Purple/violet for peak distance
  }

  // Keep the plot budget stable across window sizes. uPlot still fills the
  // available width, but a larger window no longer asks the worker to draw more samples.
  const ZOOM_THRESHOLD = 0.1

  const resetZoom = useCallback(() => {
    setZoomRange(null)
    lastRangeRef.current = null
    pendingRangeRef.current = null
    zoomRangeRef.current = null
    if (zoomStateFrameRef.current != null) {
      cancelAnimationFrame(zoomStateFrameRef.current)
      zoomStateFrameRef.current = null
    }
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
      debounceTimerRef.current = null
    }
  }, [])

  const dispatchRangeRequest = useCallback((worker: Worker, range: WorkerRange | null) => {
    const requestId = `range-${requestCounterRef.current++}`
    latestRequestIdRef.current = requestId

    // For very narrow windows (<= 200 ms span), request the full slice with no decimation
    const spanMinutes = range ? Math.max(0, range.max - range.min) : null
    const spanMs = spanMinutes != null ? spanMinutes * 60 * 1000 : null
    const effectiveTarget = spanMs != null && spanMs <= 200
      ? Number.MAX_SAFE_INTEGER
      : TARGET_POINTS * RANGE_OVERSCAN_MULTIPLIER

    worker.postMessage({
      type: 'GET_RANGE',
      requestId,
      payload: {
        range,
        targetPoints: effectiveTarget,
        dynamicDownsampling,
        zoomThreshold: ZOOM_THRESHOLD,
      },
    })
  }, [dynamicDownsampling])

  const requestRange = useCallback((range: WorkerRange | null) => {
    const normalizedRange = range ?? null
    pendingRangeRef.current = normalizedRange
    const worker = workerRef.current
    if (!worker || !workerReady) return
    dispatchRangeRequest(worker, expandRange(normalizedRange, timeRangeRef.current))
  }, [dispatchRangeRequest, workerReady])

  useEffect(() => {
    const worker = workerRef.current
    if (!worker) return

    if (!peakProperties || !Array.isArray(peakProperties.left_ips) || !Array.isArray(peakProperties.right_ips)) {
      worker.postMessage({
        type: 'SET_PEAK_PROPERTIES',
        payload: {},
      })
      setWidthSegmentsState(null)
      return
    }

    const left = Float64Array.from(peakProperties.left_ips)
    const right = Float64Array.from(peakProperties.right_ips)
    const transferList: Transferable[] = [left.buffer, right.buffer]
    let heights: Float32Array | null = null
    if (Array.isArray(peakProperties.width_heights) && peakProperties.width_heights.length > 0) {
      heights = Float32Array.from(peakProperties.width_heights)
      transferList.push(heights.buffer)
    }

    worker.postMessage(
      {
        type: 'SET_PEAK_PROPERTIES',
        payload: {
          leftIps: left,
          rightIps: right,
          widthHeights: heights,
        },
      },
      transferList
    )
    // After updating peak properties, request an updated range so width segments
    // are recomputed for the current zoom without requiring manual zooming.
    const currentRange = zoomRangeRef.current ?? zoomRange ?? null
    requestRange(currentRange)
  }, [peakProperties, requestRange, zoomRange])

  const sendDatasetToWorker = useCallback((dataset: {
    time: NumericArray
    amplitude: NumericArray
    filtered?: NumericArray | null
    totalPoints?: number
    timeRange?: { min: number; max: number } | null
  }) => {
    const worker = workerRef.current
    if (!worker) return
    setWidthSegmentsState(null)

    const timeArray = dataset.time instanceof Float64Array
      ? dataset.time.slice()
      : Float64Array.from(dataset.time as ArrayLike<number>)
    const amplitudeArray = dataset.amplitude instanceof Float32Array
      ? dataset.amplitude.slice()
      : Float32Array.from(dataset.amplitude as ArrayLike<number>)
    const transferList: Transferable[] = [timeArray.buffer, amplitudeArray.buffer]
    let filteredArray: Float32Array | null = null

    if (dataset.filtered && dataset.filtered.length > 0) {
      filteredArray = dataset.filtered instanceof Float32Array
        ? dataset.filtered.slice()
        : Float32Array.from(dataset.filtered as ArrayLike<number>)
      transferList.push(filteredArray.buffer)
    }

    const defaultRange = timeArray.length > 0
      ? { min: timeArray[0], max: timeArray[timeArray.length - 1] }
      : null

    const sourceRange = dataset.timeRange && Number.isFinite(dataset.timeRange.min) && Number.isFinite(dataset.timeRange.max)
      ? dataset.timeRange
      : defaultRange

    const convertedRange = sourceRange
      ? { min: sourceRange.min / 60, max: sourceRange.max / 60 }
      : null

    pendingRangeRef.current = zoomRangeRef.current
    setWorkerReady(false)

    worker.postMessage(
      {
        type: 'SET_DATA',
        payload: {
          time: timeArray,
          raw: amplitudeArray,
          filtered: filteredArray,
          totalPoints: dataset.totalPoints ?? dataset.time.length,
          timeRange: convertedRange,
          timeUnit: 'seconds',
        },
      },
      transferList
    )
  }, [])

  // Spin up the data worker once and listen for range updates so the main thread
  // only receives already-downsampled windows.
  useEffect(() => {
    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL("../../workers/dataWorker.ts", import.meta.url)
      )
    }

    const worker = workerRef.current
    if (!worker) {
      return
    }

    const handleMessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data
      switch (message.type) {
        case "READY": {
          setWorkerReady(true)
          setInfo((prev: any) => ({
            ...prev,
            total_points: message.payload.totalPoints,
            time_range: message.payload.timeRange,
          }))
          timeRangeRef.current = message.payload.timeRange
          const targetRange = pendingRangeRef.current ?? (dynamicDownsampling ? zoomRange : null)
          dispatchRangeRequest(worker, expandRange(targetRange ?? null, timeRangeRef.current))
          break
        }
        case "RANGE_DATA": {
          if (message.requestId && latestRequestIdRef.current && message.requestId !== latestRequestIdRef.current) {
            return
          }

          setXData(message.payload.time)
          setYOriginal(message.payload.raw)
          setYFiltered(message.payload.filtered)
          if (message.payload.widthSegments) {
            setWidthSegmentsState({
              x0: message.payload.widthSegments.x0,
              x1: message.payload.widthSegments.x1,
              y: message.payload.widthSegments.y,
            })
          } else {
            setWidthSegmentsState(null)
          }
          setInfo((prev: any) => ({
            ...prev,
            displayed_points: message.payload.displayedPoints,
            decimated: message.payload.decimated,
            visible_range: message.payload.visibleRange,
            total_points: message.payload.totalPoints ?? prev?.total_points,
          }))
          break
        }
        case "ERROR": {
          console.error("Data worker error:", message.payload.message)
          break
        }
        default:
          break
      }
    }

    worker.addEventListener("message", handleMessage)
    return () => {
      worker.removeEventListener("message", handleMessage)
    }
  }, [dispatchRangeRequest, dynamicDownsampling, zoomRange])

  useEffect(() => {
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate()
        workerRef.current = null
      }
    }
  }, [])

  // Load the full dataset once, then let the worker return downsampled display windows.
  useEffect(() => {
    if (!resultId) return
    let active = true

    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const baseBuffer = await apiClient.getResultBinary(resultId, { timeoutMs: 300000 })
        if (!active) return
        const base = parseResultBinary(baseBuffer)
        let filtered: ReturnType<typeof parseResultBinary> | null = null
        if (filteredResultId) {
          const filteredBuffer = await apiClient.getResultBinary(filteredResultId, { timeoutMs: 300000 })
          if (!active) return
          filtered = parseResultBinary(filteredBuffer)
        }
        if (!active) return

        resetZoom()
        const rangeSeconds = base.time.length > 0
          ? { min: base.time[0], max: base.time[base.time.length - 1] }
          : null
        const rangeMinutes = rangeSeconds
          ? { min: rangeSeconds.min / 60, max: rangeSeconds.max / 60 }
          : null
        setInfo({
          original_points: base.time.length,
          displayed_points: 0,
          decimated: base.time.length > TARGET_POINTS,
          time_range: rangeMinutes,
        })
        timeRangeRef.current = rangeMinutes

        sendDatasetToWorker({
          time: base.time,
          amplitude: base.amplitude,
          filtered: filtered ? filtered.amplitude : null,
          totalPoints: base.time.length,
          timeRange: rangeSeconds || undefined,
        })
      } catch (e: any) {
        if (!active) return
        console.error(e)
        setError(e?.message || "Failed to load data")
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    fetchData()
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resultId, filteredResultId])

  const handleZoomChange = useCallback((range: { min: number; max: number }) => {
    if (!workerReady) return

    const rangeChanged = !lastRangeRef.current ||
      Math.abs(lastRangeRef.current.min - range.min) > 0.001 ||
      Math.abs(lastRangeRef.current.max - range.max) > 0.001

    if (!rangeChanged) return

    lastRangeRef.current = range
    zoomRangeRef.current = range
    if (zoomStateFrameRef.current != null) {
      cancelAnimationFrame(zoomStateFrameRef.current)
    }
    zoomStateFrameRef.current = requestAnimationFrame(() => {
      zoomStateFrameRef.current = null
      setZoomRange(zoomRangeRef.current)
    })

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    // Keep data-window refresh off the hot pointer path; uPlot pans immediately
    // within the buffered range, then the worker catches up after input settles.
    debounceTimerRef.current = setTimeout(() => {
      requestRange(range)
    }, 80)
  }, [requestRange, workerReady])

  // Cleanup debounce timer
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
      if (zoomStateFrameRef.current != null) {
        cancelAnimationFrame(zoomStateFrameRef.current)
        zoomStateFrameRef.current = null
      }
    }
  }, [])

  const summary = useMemo(() => {
    const totalPeaks = peakTimes?.length || 0
    if (!totalPeaks) return null

    const widthScale = typeof timeResolution === "number" && Number.isFinite(timeResolution)
      ? timeResolution * 1000
      : 1
    const widths = cleanNumbers(peakProperties?.widths || []).map((w) => w * widthScale)
    const avgWidth = mean(widths)

    const areaVals: number[] = []
    let totalArea = 0
    const pairs = Math.min(peakAmplitudes.length, widths.length)
    for (let i = 0; i < pairs; i++) {
      const amp = peakAmplitudes[i]
      const width = widths[i]
      if (Number.isFinite(amp) && Number.isFinite(width)) {
        // Convert width from ms to seconds so area is in "counts" (amplitude × seconds)
        const widthSeconds = width / 1000
        const area = amp * widthSeconds
        areaVals.push(area)
        totalArea += area
      }
    }
    const avgArea = mean(areaVals)

    const throughputVals: number[] = []
    for (let i = 0; i < peakTimes.length - 1; i++) {
      const dt = peakTimes[i + 1] - peakTimes[i]
      if (Number.isFinite(dt) && dt > 0) {
        throughputVals.push(1 / dt)
      }
    }
    const avgThroughput = mean(throughputVals)

    const prominences = cleanNumbers(peakProperties?.prominences || [])
    const avgProminence = mean(prominences)

    // Compute peak area as (total area within peaks) per second of recording
    let peakAreaPerSecond: number | null = null
    if (areaVals.length > 0 && peakTimes.length > 1) {
      const totalDurationSeconds = peakTimes[peakTimes.length - 1] - peakTimes[0]
      if (Number.isFinite(totalDurationSeconds) && totalDurationSeconds > 0) {
        peakAreaPerSecond = totalArea / totalDurationSeconds
      }
    }

    return {
      count: totalPeaks,
      meanProminence: avgProminence,
      meanWidth: avgWidth,
      meanThroughput: avgThroughput,
      peakAreaPerSecond,
    }
  }, [peakTimes, peakAmplitudes, peakProperties, timeResolution])

  const widthsMs = useMemo(() => {
    return (peakProperties?.widths || []).map((w: number) => w * (
      typeof timeResolution === "number" && Number.isFinite(timeResolution) ? timeResolution * 1000 : 1
    ))
  }, [peakProperties?.widths, timeResolution])

  // Calculate vertical lines for histograms
  const histogramVerticalLines = useMemo(() => {
    const lines: {
      amplitude: Array<{ value: number; color?: string; label?: string }>
      width: Array<{ value: number; color?: string; label?: string }>
      interval: Array<{ value: number; color?: string; label?: string }>
    } = {
      amplitude: [],
      width: [],
      interval: [],
    }

    // Accent color for threshold lines
    const thresholdColor = isDark ? "#fb923c" : "#f97316"
    
    // Amplitude histogram: prominence threshold
    if (prominenceThreshold != null && Number.isFinite(prominenceThreshold)) {
      lines.amplitude.push({
        value: prominenceThreshold,
        color: thresholdColor,
        label: "Threshold",
      })
    }

    // Width histogram: min and max width in ms
    if (widthMs) {
      const [minStr, maxStr] = widthMs.split(',')
      const minWidth = parseFloat(minStr)
      const maxWidth = parseFloat(maxStr)

      if (Number.isFinite(minWidth)) {
        lines.width.push({
          value: minWidth,
          color: thresholdColor,
          label: "Min",
        })
      }

      if (Number.isFinite(maxWidth)) {
        lines.width.push({
          value: maxWidth,
          color: thresholdColor,
          label: "Max",
        })
      }
    }

    // Interval histogram: distance in ms
    if (distance != null && Number.isFinite(distance) && timeResolution != null && Number.isFinite(timeResolution)) {
      const distanceMs = distance * timeResolution * 1000
      lines.interval.push({
        value: distanceMs,
        color: thresholdColor,
        label: "Min Distance",
      })
    }

    return lines
  }, [prominenceThreshold, distance, widthMs, timeResolution, isDark])

  // Fetch histogram data when peaks are detected
  useEffect(() => {
    const prominenceValues = cleanNumbers(peakProperties?.prominences || [])

    if (prominenceValues.length === 0) {
      setHistogramData(null)
      return
    }

    const fetchHistogram = async () => {
      setHistogramLoading(true)
      try {
        const intervalValues = peakIntervals || []

        const response = await apiClient.generateHistograms(
          {
            peakAmplitudes: prominenceValues,
            peakWidthsMs: widthsMs,
            peakIntervalsMs: intervalValues,
          },
          {
            binCount: histogramConfig.binCount,
            metrics: histogramScales,
          }
        )

        // Validate histogram data before setting
        const isValidHistogram = (hist: any) => {
          return hist &&
            Array.isArray(hist.bins) &&
            Array.isArray(hist.counts) &&
            hist.bins.length > 0 &&
            hist.counts.length > 0 &&
            hist.bins.every((v: any) => Number.isFinite(v)) &&
            hist.counts.every((v: any) => Number.isFinite(v))
        }

        const validatedResponse = {
          amplitude: isValidHistogram(response.amplitude) ? response.amplitude : { bins: [], counts: [] },
          width: isValidHistogram(response.width) ? response.width : { bins: [], counts: [] },
          interval: isValidHistogram(response.interval) ? response.interval : { bins: [], counts: [] },
        }

        setHistogramData(validatedResponse)
      } catch (error) {
        console.error("Failed to generate histograms:", error)
        setHistogramData(null)
      } finally {
        setHistogramLoading(false)
      }
    }

    fetchHistogram()
  }, [peakProperties, widthsMs, peakIntervals, histogramScales, histogramConfig])

  const formatStat = (value: number | null | undefined, digits = 2) => {
    if (value == null || !Number.isFinite(value)) return "—"
    return value.toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: value < 10 ? Math.min(1, digits) : 0,
    })
  }

  const formatZoomRange = (range: { min: number; max: number } | null) => {
    if (!range || !Number.isFinite(range.min) || !Number.isFinite(range.max)) return ""
    const spanMin = Math.max(0, range.max - range.min)
    if (spanMin <= (1 / 60)) {
      // < 1 second: show ms
      const lo = Math.round(range.min * 60000)
      const hi = Math.round(range.max * 60000)
      return `${lo}–${hi} ms`
    }
    if (spanMin < 1) {
      // < 1 minute: show seconds
      const spanSec = spanMin * 60
      const decimals = spanSec >= 5 ? 0 : 1
      const lo = (range.min * 60).toFixed(decimals)
      const hi = (range.max * 60).toFixed(decimals)
      return `${lo}–${hi} s`
    }
    // minutes
    const decimals = spanMin >= 10 ? 0 : 2
    return `${range.min.toFixed(decimals)}–${range.max.toFixed(decimals)} min`
  }

  const widthSegments = useMemo(() => {
    if (!widthSegmentsState) return null
    const { x0, x1, y } = widthSegmentsState
    const count = Math.min(x0.length, x1.length, y.length)
    const segments: { x0: number; x1: number; y: number; width: number }[] = []
    for (let i = 0; i < count; i += 1) {
      // Calculate width in milliseconds (x values are in minutes)
      const widthMs = (x1[i] - x0[i]) * 60 * 1000
      segments.push({ x0: x0[i], x1: x1[i], y: y[i], width: widthMs })
    }
    return segments
  }, [widthSegmentsState])

  // Recalculate series when data or peaks change
  // NOTE: Must be called before any early returns to satisfy Rules of Hooks
  const series = useMemo(() => {
    const baseSeries = [
      {
        label: "Original",
        color: isDark ? "rgba(148, 163, 184, 0.5)" : "rgba(100, 116, 139, 0.5)",
        width: 1,
        data: yOriginal,
      },
      ...(yFiltered ? [{
        label: "Filtered",
        color: isDark ? "rgba(220, 220, 220, 0.9)" : "rgba(40, 40, 40, 0.85)",
        width: 1.2,
        data: yFiltered,
      }] : []),
    ]

    // Peaks overlay as point series (sparse)
    // Convert peak times to minutes and map to current xData indices
    const peakSeries = (peakTimes?.length || 0) > 0 ? (() => {
      const peakYVals = peakAmplitudes || []

      // Get visible range to filter peaks
      const visibleMin = xData.length > 0 ? xData[0] : 0
      const visibleMax = xData.length > 0 ? xData[xData.length - 1] : 0

      const sparse = new Array(xData.length).fill(null) as (number | null)[]
      const startPeak = lowerBound(peakTimes, visibleMin * 60)
      const endPeak = upperBound(peakTimes, visibleMax * 60)
      const peakIndices: number[] = []

      for (let i = startPeak; i < endPeak; i += 1) {
        const px = Math.max(0, peakTimes[i] / 60)
        // Only show peaks in visible range
        if (px < visibleMin || px > visibleMax) continue

        // Find nearest index in xData using binary search
        let idx = -1
        let lo = 0, hi = xData.length - 1
        while (lo <= hi) {
          const mid = (lo + hi) >> 1
          if (xData[mid] < px) lo = mid + 1
          else hi = mid - 1
        }
        // lo is first >= px; choose closer between lo and lo-1
        const candidates = [lo, lo - 1].filter(j => j >= 0 && j < xData.length)
        if (candidates.length) {
          idx = candidates.reduce((best, cur) =>
            Math.abs(xData[cur] - px) < Math.abs(xData[best] - px) ? cur : best,
            candidates[0]
          )
        }
        if (idx >= 0 && idx < sparse.length) {
          sparse[idx] = peakYVals[i]
          peakIndices.push(idx)
        }
      }
      return [{
        label: "Peaks",
        color: isDark ? "#f87171" : "#ef4444",
        width: 0,
        data: sparse as any,
        points: true,
        pointSize: 4,
        dataIndices: peakIndices,
      }]
    })() : []

    return [...baseSeries, ...peakSeries]
  }, [yOriginal, yFiltered, xData, peakTimes, peakAmplitudes, isDark])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-3">
          <Loader2 className="h-8 w-8 animate-spin mx-auto text-accent" />
          <p className="text-sm text-muted-foreground">Loading data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-2">
          <p className="text-sm text-destructive">{error}</p>
        </div>
      </div>
    )
  }

  if (!xData.length) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-sm text-muted-foreground">No data available</p>
      </div>
    )
  }

  return (
    <div className={`flex flex-col w-full ${className}`}>
      {/* Dataset info */}
      <div className="px-4 lg:px-8 pt-4 lg:pt-6 pb-2 shrink-0">
        {info && (
          <p className="text-[11px] text-muted-foreground mb-1">
            {info.original_points?.toLocaleString()} points
            {info.displayed_points && (
              <> • displaying {info.displayed_points.toLocaleString()}</>
            )}
            {info.decimated && (
              <> <span className="text-muted-foreground/50">(target: {TARGET_POINTS.toLocaleString()})</span></>
            )}
            {dynamicDownsampling && zoomRange && (
              <> • zoomed: {formatZoomRange(zoomRange)}</>
            )}
            {" • "}
            <span className="text-muted-foreground/70">Scroll to zoom • Double-click to reset</span>
          </p>
        )}
      </div>

      {/* Chart and Statistics side-by-side */}
      <div className="flex flex-col xl:flex-row gap-4 px-4 pb-6 min-w-0">
        {/* uPlot chart */}
        <div className="flex-1 min-w-0" style={{ minWidth: 300 }}>
          <UPlotChart
            xData={xData}
            series={series}
            xLabel="Time (min)"
            yLabel="Counts"
            height={400}
            widthSegments={widthSegments || undefined}
            xRange={zoomRange || undefined} // Preserve zoom when data updates
            xBounds={info?.time_range || undefined}
            onXRangeChange={handleZoomChange}
            onResetZoom={() => {
              resetZoom()
              requestRange(null)
            }}
          />
        </div>

        {/* Peak Statistics - Vertical box on the right, stacks below chart on smaller screens */}
        {summary && (
          <div className="w-full xl:w-[200px] flex-shrink-0 min-w-0">
            <div className="rounded-lg border bg-card/60 shadow-sm h-full">
              <div className="flex items-center justify-between border-b px-3 py-2">
                <div className="truncate">
                  <p className="text-xs font-semibold">Peak Statistics</p>
                </div>
              </div>
              <div className="p-3 space-y-3">
                {/* Number of peaks */}
                <div className="space-y-1">
                  <p className="text-[11px] text-muted-foreground">Detected peaks</p>
                  <p className="text-sm font-semibold">{summary.count.toLocaleString()}</p>
                </div>

                {/* Mean Prominence */}
                <div className="space-y-1">
                  <p className="text-[11px] text-muted-foreground">Mean prominence</p>
                  <p className="text-sm font-semibold">{formatStat(summary.meanProminence, 2)}</p>
                </div>

                {/* Mean Width */}
                <div className="space-y-1">
                  <p className="text-[11px] text-muted-foreground">Mean peak width</p>
                  <p className="text-sm font-semibold">
                    {formatStat(summary.meanWidth, 3)}{" "}
                    <span className="text-[10px] text-muted-foreground">ms</span>
                  </p>
                </div>

                {/* Mean Throughput */}
                <div className="space-y-1">
                  <p className="text-[11px] text-muted-foreground">Mean throughput</p>
                  <p className="text-sm font-semibold">
                    {formatStat(summary.meanThroughput, 2)}{" "}
                    <span className="text-[10px] text-muted-foreground">peaks/s</span>
                  </p>
                </div>

                {/* Peak area (counts per second) */}
                <div className="space-y-1">
                  <p className="text-[11px] text-muted-foreground">Peak area</p>
                  <p className="text-sm font-semibold">
                    {formatStat(summary.peakAreaPerSecond, 2)}{" "}
                    <span className="text-[10px] text-muted-foreground">counts/s</span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Histograms */}
      <div className="px-4 lg:px-6 pb-6 space-y-4">
        {peakAmplitudes.length > 0 && (
          <>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-semibold">Peak Distributions</h3>
                <p className="text-xs text-muted-foreground">Statistical analysis of detected peaks</p>
              </div>
              <div className="flex items-center gap-3 bg-muted/30 px-3 py-1.5 rounded-md border self-start sm:self-auto">
                <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Bins</span>
                <input
                  type="range"
                  min={20}
                  max={200}
                  step={10}
                  value={histogramConfig.binCount}
                  onChange={(event) => updateBinCount(Number(event.target.value))}
                  className="w-24 h-1.5 rounded-lg cursor-pointer accent-accent"
                  style={{ accentColor: accentColor }}
                />
                <span className="text-xs font-mono w-6 text-right">{histogramConfig.binCount}</span>
              </div>
            </div>

            {histogramLoading ? (
              <div className="flex items-center justify-center py-12 border rounded-xl bg-card/50">
                <div className="text-center space-y-3">
                  <Loader2 className="h-8 w-8 animate-spin text-accent mx-auto" />
                  <p className="text-sm text-muted-foreground">Calculating distributions...</p>
                </div>
              </div>
            ) : histogramData ? (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Prominence Histogram */}
                <HistogramCard
                  title="Prominence"
                  data={histogramData.amplitude}
                  color={histogramColors.amplitude}
                  xLabel="Prominence"
                  yLabel="Count"
                  xScale={histogramScales.amplitude.xScale}
                  yScale={histogramScales.amplitude.yScale}
                  onXScaleChange={(val) => updateXAxisScale("amplitude", val)}
                  onYScaleChange={(val) => updateYAxisScale("amplitude", val)}
                  verticalLines={histogramVerticalLines.amplitude}
                />

                {/* Width Histogram */}
                <HistogramCard
                  title="Peak Width"
                  data={histogramData.width}
                  color={histogramColors.width}
                  xLabel="Width (ms)"
                  yLabel="Count"
                  xScale={histogramScales.width.xScale}
                  yScale={histogramScales.width.yScale}
                  onXScaleChange={(val) => updateXAxisScale("width", val)}
                  onYScaleChange={(val) => updateYAxisScale("width", val)}
                  verticalLines={histogramVerticalLines.width}
                />

                {/* Interval Histogram */}
                <HistogramCard
                  title="Peak Distance"
                  data={histogramData.interval}
                  color={histogramColors.interval}
                  xLabel="Distance (ms)"
                  yLabel="Count"
                  xScale={histogramScales.interval.xScale}
                  yScale={histogramScales.interval.yScale}
                  onXScaleChange={(val) => updateXAxisScale("interval", val)}
                  onYScaleChange={(val) => updateYAxisScale("interval", val)}
                  verticalLines={histogramVerticalLines.interval}
                />
              </div>
            ) : (
              <div className="py-12 text-sm text-muted-foreground text-center border rounded-xl bg-card/50">
                <p>Generating histograms...</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
