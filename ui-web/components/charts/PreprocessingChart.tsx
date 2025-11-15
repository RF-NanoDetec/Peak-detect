"use client"

import { useEffect, useMemo, useState, useRef, useCallback } from "react"
import { Loader2 } from "lucide-react"
import { apiClient } from "@/lib/apiClient"
import { UPlotChart } from "./UPlotChart"
import { useTheme } from "next-themes"
import type { DataPreviewResponse } from "@/lib/types"
import { parseResultBinary } from "@/lib/binaryParsers"

interface PreprocessingChartProps {
  resultId: string
  filteredResultId?: string | null
  className?: string
  timeResolution?: number
  peakTimes?: number[]
  peakAmplitudes?: number[]
  peakProperties?: {
    prominences?: number[]
    left_ips: number[]
    right_ips: number[]
    widths: number[]
    width_heights?: number[]
  } | null
  previewData?: DataPreviewResponse | null
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

export function PreprocessingChart({ 
  resultId, 
  filteredResultId, 
  className = "",
  timeResolution,
  peakTimes = [],
  peakAmplitudes = [],
  peakProperties = null,
  previewData = null,
}: PreprocessingChartProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<any>(null)
  
  // Displayed data (fed by worker)
  const [xData, setXData] = useState<NumericArray>([])
  const [yOriginal, setYOriginal] = useState<NumericArray>([])
  const [yFiltered, setYFiltered] = useState<NumericArray | null>(null)
  
  const [widthSegmentsState, setWidthSegmentsState] = useState<{
    x0: Float64Array
    x1: Float64Array
    y: Float32Array
  } | null>(null)
  
  const [fullRes, setFullRes] = useState<boolean>(false)
  const [zoomRange, setZoomRange] = useState<WorkerRange>(null)
  const [dynamicDownsampling, setDynamicDownsampling] = useState<boolean>(true)
  const [workerReady, setWorkerReady] = useState(false)
  
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const lastRangeRef = useRef<WorkerRange>(null)
  const pendingRangeRef = useRef<WorkerRange>(null)
  const zoomRangeRef = useRef<WorkerRange>(null)
  const latestRequestIdRef = useRef<string | null>(null)
  const requestCounterRef = useRef(0)
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === "dark"
  
  const TARGET_POINTS = 10000 // Target number of points to display
  const ZOOM_THRESHOLD = 0.1

  const resetZoom = useCallback(() => {
    setZoomRange(null)
    lastRangeRef.current = null
    pendingRangeRef.current = null
    zoomRangeRef.current = null
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
      debounceTimerRef.current = null
    }
  }, [])

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
  }, [peakProperties])

  const dispatchRangeRequest = useCallback((worker: Worker, range: WorkerRange | null) => {
    const requestId = `range-${requestCounterRef.current++}`
    latestRequestIdRef.current = requestId
    worker.postMessage({
      type: 'GET_RANGE',
      requestId,
      payload: {
        range,
        targetPoints: TARGET_POINTS,
        dynamicDownsampling,
        zoomThreshold: ZOOM_THRESHOLD,
      },
    })
  }, [TARGET_POINTS, dynamicDownsampling])

  const requestRange = useCallback((range: WorkerRange | null) => {
    const normalizedRange = range ?? null
    pendingRangeRef.current = normalizedRange
    const worker = workerRef.current
    if (!worker || !workerReady) return
    dispatchRangeRequest(worker, normalizedRange)
  }, [dispatchRangeRequest, workerReady])

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
      workerRef.current = new Worker(new URL("../../workers/dataWorker.ts", import.meta.url), { type: "module" })
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
          const targetRange = pendingRangeRef.current ?? (dynamicDownsampling ? zoomRange : null)
          dispatchRangeRequest(worker, targetRange ?? null)
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

  // Load data (preview by default) and hydrate the worker
  useEffect(() => {
    if (!resultId) return
    let active = true

    const applyPreviewResponse = (response: DataPreviewResponse) => {
      if (!active) return
      resetZoom()
      const previewRange = response.time.length > 0
        ? { min: response.time[0] / 60, max: response.time[response.time.length - 1] / 60 }
        : null
      setInfo({
        original_points: response.total_points,
        displayed_points: response.time.length,
        decimated: response.decimated,
        time_range: previewRange,
      })
      sendDatasetToWorker({
        time: response.time,
        amplitude: response.amplitude,
        filtered: response.filtered_amplitude || null,
        totalPoints: response.total_points,
        timeRange: response.time.length > 0
          ? { min: response.time[0], max: response.time[response.time.length - 1] }
          : undefined,
      })
    }

    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        if (!fullRes && !filteredResultId && previewData) {
          applyPreviewResponse(previewData)
          setLoading(false)
          return
        }

        if (fullRes) {
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

          sendDatasetToWorker({
            time: base.time,
            amplitude: base.amplitude,
            filtered: filtered ? filtered.amplitude : null,
            totalPoints: base.time.length,
            timeRange: rangeSeconds || undefined,
          })
        } else {
          const response = await apiClient.getDataPreview(resultId, filteredResultId || undefined, { full: false })
          if (!active) return
          applyPreviewResponse(response)
        }
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
  }, [resultId, filteredResultId, fullRes])

  const handleZoomChange = useCallback((range: { min: number; max: number }) => {
    if (!workerReady) return

    const rangeChanged = !lastRangeRef.current ||
      Math.abs(lastRangeRef.current.min - range.min) > 0.001 ||
      Math.abs(lastRangeRef.current.max - range.max) > 0.001

    if (!rangeChanged) return

    lastRangeRef.current = range
    setZoomRange(range)
    zoomRangeRef.current = range

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    debounceTimerRef.current = setTimeout(() => {
      requestRange(range)
    }, 100)
  }, [requestRange, workerReady])

  // Cleanup debounce timer
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
    }
  }, [])

  const cleanNumbers = (values: number[] = []) => values.filter((v) => Number.isFinite(v))
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

  const summary = useMemo(() => {
    const totalPeaks = peakTimes?.length || 0
    if (!totalPeaks) return null

    const avgAmplitude = mean(peakAmplitudes)

    const widthScale = typeof timeResolution === "number" && Number.isFinite(timeResolution)
      ? timeResolution * 1000
      : 1
    const widths = cleanNumbers(peakProperties?.widths || []).map((w) => w * widthScale)
    const avgWidth = mean(widths)

    const areaVals: number[] = []
    const pairs = Math.min(peakAmplitudes.length, widths.length)
    for (let i = 0; i < pairs; i++) {
      const amp = peakAmplitudes[i]
      const width = widths[i]
      if (Number.isFinite(amp) && Number.isFinite(width)) {
        areaVals.push(amp * width)
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
    const prominenceStd = std(prominences)
    const snrRatio = avgProminence && prominenceStd ? avgProminence / prominenceStd : null
    const snrDb = snrRatio && snrRatio > 0 ? 20 * Math.log10(snrRatio) : null

    return {
      count: totalPeaks,
      meanAmplitude: avgAmplitude,
      meanWidth: avgWidth,
      meanArea: avgArea,
      meanThroughput: avgThroughput,
      snrRatio,
      snrDb,
    }
  }, [peakTimes, peakAmplitudes, peakProperties, timeResolution])

  const formatStat = (value: number | null | undefined, digits = 2) => {
    if (value == null || !Number.isFinite(value)) return "—"
    return value.toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: value < 10 ? Math.min(1, digits) : 0,
    })
  }

  const widthSegments = useMemo(() => {
    if (!widthSegmentsState) return null
    const { x0, x1, y } = widthSegmentsState
    const count = Math.min(x0.length, x1.length, y.length)
    const segments: { x0: number; x1: number; y: number }[] = []
    for (let i = 0; i < count; i += 1) {
      segments.push({ x0: x0[i], x1: x1[i], y: y[i] })
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
        color: isDark ? "#60a5fa" : "#3b82f6",
        width: 1.5,
        data: yFiltered,
      }] : []),
    ]

    // Peaks overlay as point series (sparse)
    // Convert peak times to minutes and map to current xData indices
    const peakSeries = (peakTimes?.length || 0) > 0 ? (() => {
      const peakXMin = (peakTimes || []).map(t => Math.max(0, t / 60))
      const peakYVals = peakAmplitudes || []
      
      // Get visible range to filter peaks
      const visibleMin = xData.length > 0 ? xData[0] : 0
      const visibleMax = xData.length > 0 ? xData[xData.length - 1] : 0
      
      const sparse = new Array(xData.length).fill(null) as (number | null)[]
      peakXMin.forEach((px, i) => {
        // Only show peaks in visible range
        if (px < visibleMin || px > visibleMax) return
        
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
        }
      })
      return [{
        label: "Peaks",
        color: isDark ? "#f87171" : "#ef4444",
        width: 0,
        data: sparse as any,
        points: true,
        pointSize: 6,
      }]
    })() : []

    return [...baseSeries, ...peakSeries]
  }, [yOriginal, yFiltered, xData, peakTimes, peakAmplitudes, isDark])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-3">
          <Loader2 className="h-8 w-8 animate-spin mx-auto text-primary" />
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
    <div className={`flex flex-col h-full w-full ${className}`} style={{ minHeight: 0 }}>
      {/* Dataset info + controls */}
      <div className="px-8 pt-6 pb-2 flex items-center justify-between flex-shrink-0">
        <div>
          {info && (
            <p className="text-[11px] text-muted-foreground">
              {info.original_points?.toLocaleString()} points
              {info.displayed_points && (
                <> • displaying {info.displayed_points.toLocaleString()}</>
              )}
              {dynamicDownsampling && fullRes && zoomRange && (
                <> • zoomed: {zoomRange.min.toFixed(2)}–{zoomRange.max.toFixed(2)} min</>
              )}
              {" • "}
              <span className="text-muted-foreground/70">Scroll to zoom • Drag to pan • Double-click to reset</span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-4">
          <label className="text-[11px] text-muted-foreground flex items-center gap-2">
            <input
              type="checkbox"
              className="h-3 w-3"
              checked={!!fullRes}
              onChange={(e) => setFullRes(e.target.checked)}
              disabled={loading}
            />
            Load full-resolution data
          </label>
          {fullRes && (
            <label className="text-[11px] text-muted-foreground flex items-center gap-2">
              <input
                type="checkbox"
                className="h-3 w-3"
                checked={dynamicDownsampling}
              onChange={(e) => {
                const enabled = e.target.checked
                setDynamicDownsampling(enabled)
                if (!enabled) {
                  resetZoom()
                  requestRange(null)
                } else {
                  requestRange(zoomRange)
                }
              }}
                disabled={loading}
              />
              Dynamic downsampling (zoom-adaptive)
            </label>
          )}
        </div>
      </div>
      
      {/* uPlot chart */}
      <div className="flex-1 px-4 pb-6" style={{ minHeight: 0 }}>
        <UPlotChart
          xData={xData}
          series={series}
          xLabel="Time (min)"
          yLabel="Counts"
          height={400}
          widthSegments={widthSegments || undefined}
          xRange={zoomRange || undefined} // Preserve zoom when data updates
          onXRangeChange={handleZoomChange}
          onResetZoom={() => {
            resetZoom()
            requestRange(null)
          }}
        />
      </div>

      <div className="px-6 pb-6 flex-shrink-0">
        <div className="rounded-xl border bg-card/60 shadow-sm">
          <div className="flex items-center justify-between border-b px-4 py-3">
            <div>
              <p className="text-sm font-semibold">Peak Detection Summary</p>
              <p className="text-xs text-muted-foreground">
                {summary ? 'Statistics from the latest detection run' : 'Detect peaks to populate these statistics'}
              </p>
            </div>
          </div>
          {summary ? (
            <dl className="grid gap-4 p-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
              <div>
                <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Detected Peaks</dt>
                <dd className="text-2xl font-semibold">{summary.count.toLocaleString()}</dd>
                <p className="text-xs text-muted-foreground mt-1">Total peaks above thresholds</p>
              </div>
              <div>
                <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Mean Amplitude</dt>
                <dd className="text-xl font-semibold">
                  {formatStat(summary.meanAmplitude, 2)} <span className="text-xs text-muted-foreground">a.u.</span>
                </dd>
                <p className="text-xs text-muted-foreground mt-1">Average peak height</p>
              </div>
              <div>
                <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Mean Width</dt>
                <dd className="text-xl font-semibold">
                  {formatStat(summary.meanWidth, 2)} <span className="text-xs text-muted-foreground">ms</span>
                </dd>
                <p className="text-xs text-muted-foreground mt-1">Measured at half prominence</p>
              </div>
              <div>
                <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Mean Area</dt>
                <dd className="text-xl font-semibold">
                  {formatStat(summary.meanArea, 2)} <span className="text-xs text-muted-foreground">a.u.·ms</span>
                </dd>
                <p className="text-xs text-muted-foreground mt-1">Amplitude × width approximation</p>
              </div>
              <div>
                <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Mean Throughput</dt>
                <dd className="text-xl font-semibold">
                  {formatStat(summary.meanThroughput, 2)} <span className="text-xs text-muted-foreground">peaks/s</span>
                </dd>
                <p className="text-xs text-muted-foreground mt-1">Averaged reciprocal spacing</p>
              </div>
              <div>
                <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Estimated SNR</dt>
                <dd className="text-xl font-semibold">
                  {summary.snrRatio && Number.isFinite(summary.snrRatio)
                    ? `${formatStat(summary.snrRatio, 2)} : 1`
                    : '—'}
                </dd>
                <p className="text-xs text-muted-foreground mt-1">
                  {summary.snrDb && Number.isFinite(summary.snrDb)
                    ? `${formatStat(summary.snrDb, 1)} dB`
                    : 'Mean prominence vs. variation'}
                </p>
              </div>
            </dl>
          ) : (
            <div className="p-6 text-sm text-muted-foreground text-center">
              Run peak detection to see amplitude, width, throughput, and SNR statistics here.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
