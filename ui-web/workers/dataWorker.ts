/// <reference lib="webworker" />

type Range = { min: number; max: number } | null

interface BaseMessage {
  requestId?: string
}

interface SetDataMessage extends BaseMessage {
  type: 'SET_DATA'
  payload: {
    time: Float64Array
    raw: Float32Array
    filtered?: Float32Array | null
    totalPoints?: number
    timeRange?: { min: number; max: number }
    timeUnit?: 'seconds' | 'minutes'
  }
}

interface UpdateFilteredMessage extends BaseMessage {
  type: 'UPDATE_FILTERED'
  payload: {
    filtered: Float32Array | null
  }
}

interface GetRangeMessage extends BaseMessage {
  type: 'GET_RANGE'
  payload: {
    range: Range
    targetPoints: number
    dynamicDownsampling: boolean
    zoomThreshold: number
  }
}

interface SetPeakPropertiesMessage extends BaseMessage {
  type: 'SET_PEAK_PROPERTIES'
  payload: {
    leftIps?: Float64Array | null
    rightIps?: Float64Array | null
    widthHeights?: Float32Array | null
  }
}

interface ClearMessage extends BaseMessage {
  type: 'CLEAR'
}

type WorkerRequest =
  | SetDataMessage
  | UpdateFilteredMessage
  | GetRangeMessage
  | ClearMessage
  | SetPeakPropertiesMessage

type RangeResponse = {
  type: 'RANGE_DATA'
  requestId?: string
  payload: {
    time: Float64Array
    raw: Float32Array
    filtered: Float32Array | null
    displayedPoints: number
    totalPoints: number
    visibleRange: Range
    decimated: boolean
    widthSegments: {
      x0: Float64Array
      x1: Float64Array
      y: Float32Array
    } | null
  }
}

type ReadyResponse = {
  type: 'READY'
  payload: {
    totalPoints: number
    timeRange: Range
  }
}

type ErrorResponse = {
  type: 'ERROR'
  requestId?: string
  payload: { message: string }
}

type WorkerResponse = RangeResponse | ReadyResponse | ErrorResponse

const ctx: DedicatedWorkerGlobalScope = self as any

let timeData: Float64Array | null = null
let rawData: Float32Array | null = null
let filteredData: Float32Array | null = null
let peakLeftIps: Float64Array | null = null
let peakRightIps: Float64Array | null = null
let peakWidthHeights: Float32Array | null = null
let totalPoints = 0
let cachedRange: Range = null

const postMessage = (message: WorkerResponse, transfer: Transferable[] = []) => {
  ctx.postMessage(message, transfer)
}

const clampRange = (value: number, min: number, max: number) => {
  if (value < min) return min
  if (value > max) return max
  return value
}

const binarySearch = (array: Float64Array, value: number, findUpperBound: boolean) => {
  let low = 0
  let high = array.length - 1
  let result = findUpperBound ? array.length : 0

  while (low <= high) {
    const mid = low + ((high - low) >> 1)
    const midVal = array[mid]

    if (midVal === value) {
      return mid
    }

    if (midVal < value) {
      low = mid + 1
      if (!findUpperBound) {
        result = low
      }
    } else {
      high = mid - 1
      if (findUpperBound) {
        result = mid
      }
    }
  }

  return clampRange(result, 0, array.length)
}

const createSequentialIndices = (start: number, end: number) => {
  const length = Math.max(end - start, 0)
  const indices = new Uint32Array(length)
  for (let i = 0; i < length; i += 1) {
    indices[i] = start + i
  }
  return indices
}

const fractionalIndexToTime = (idx: number) => {
  if (!timeData || timeData.length === 0) {
    return 0
  }
  if (idx <= 0) {
    return timeData[0]
  }
  if (idx >= timeData.length - 1) {
    return timeData[timeData.length - 1]
  }
  const i0 = Math.floor(idx)
  const frac = idx - i0
  const i1 = Math.min(i0 + 1, timeData.length - 1)
  const t0 = timeData[i0]
  const t1 = timeData[i1]
  return t0 + frac * (t1 - t0)
}

const buildWidthSegmentsForRange = (visibleRange: Range) => {
  if (!peakLeftIps || !peakRightIps || !timeData || timeData.length === 0) {
    return null
  }

  const count = Math.min(peakLeftIps.length, peakRightIps.length)
  if (count === 0) {
    return null
  }

  const x0: number[] = []
  const x1: number[] = []
  const y: number[] = []

  for (let i = 0; i < count; i += 1) {
    const startIdx = peakLeftIps[i]
    const endIdx = peakRightIps[i]
    if (!Number.isFinite(startIdx) || !Number.isFinite(endIdx)) {
      continue
    }

    const xStart = fractionalIndexToTime(startIdx)
    const xEnd = fractionalIndexToTime(endIdx)

    if (
      visibleRange &&
      (xEnd < visibleRange.min || xStart > visibleRange.max)
    ) {
      continue
    }

    let height = 0
    if (peakWidthHeights && peakWidthHeights.length > i) {
      height = peakWidthHeights[i]
    } else {
      const midpointIdx = Math.round((startIdx + endIdx) / 2)
      if (filteredData && midpointIdx >= 0 && midpointIdx < filteredData.length) {
        height = filteredData[midpointIdx]
      } else if (rawData && midpointIdx >= 0 && midpointIdx < rawData.length) {
        height = rawData[midpointIdx]
      }
    }

    x0.push(xStart)
    x1.push(xEnd)
    y.push(height)
  }

  if (!x0.length) {
    return null
  }

  return {
    x0: Float64Array.from(x0),
    x1: Float64Array.from(x1),
    y: Float32Array.from(y),
  }
}

const buildSlicesFromIndices = (indices: Uint32Array) => {
  if (!timeData || !rawData) {
    return {
      time: new Float64Array(0),
      raw: new Float32Array(0),
      filtered: null,
    }
  }

  const timeSlice = new Float64Array(indices.length)
  const rawSlice = new Float32Array(indices.length)
  const hasFiltered = !!filteredData
  const filteredSlice = hasFiltered ? new Float32Array(indices.length) : null

  for (let i = 0; i < indices.length; i += 1) {
    const idx = indices[i]
    timeSlice[i] = timeData[idx]
    rawSlice[i] = rawData[idx]
    if (hasFiltered && filteredSlice && filteredData) {
      filteredSlice[i] = filteredData[idx]
    }
  }

  return {
    time: timeSlice,
    raw: rawSlice,
    filtered: filteredSlice,
  }
}

const decimateSegment = (startIdx: number, endIdx: number, maxPoints: number) => {
  const length = Math.max(endIdx - startIdx, 0)

  if (!timeData || !rawData || length <= 0) {
    return {
      indices: new Uint32Array(0),
      decimated: false,
    }
  }

  if (length <= maxPoints) {
    return {
      indices: createSequentialIndices(startIdx, endIdx),
      decimated: false,
    }
  }

  const nBins = Math.max(1, Math.floor(maxPoints / 2))
  const binSize = Math.max(1, Math.floor(length / nBins))
  const outIndices: number[] = []

  for (let bin = 0; bin < nBins; bin += 1) {
    const start = startIdx + bin * binSize
    const end = bin === nBins - 1 ? endIdx : Math.min(endIdx, start + binSize)

    if (end <= start) {
      continue
    }

    let localMinIdx = start
    let localMaxIdx = start
    let minVal = rawData[start]
    let maxVal = rawData[start]

    for (let i = start + 1; i < end; i += 1) {
      const value = rawData[i]
      if (value < minVal) {
        minVal = value
        localMinIdx = i
      }
      if (value > maxVal) {
        maxVal = value
        localMaxIdx = i
      }
    }

    if (localMinIdx <= localMaxIdx) {
      outIndices.push(localMinIdx, localMaxIdx)
    } else {
      outIndices.push(localMaxIdx, localMinIdx)
    }
  }

  const deduped: number[] = []
  let lastIdx = -1

  for (let i = 0; i < outIndices.length; i += 1) {
    const idx = outIndices[i]
    if (idx !== lastIdx) {
      deduped.push(idx)
      lastIdx = idx
    }
  }

  const decimatedIndices = new Uint32Array(deduped.length)
  for (let i = 0; i < deduped.length; i += 1) {
    decimatedIndices[i] = deduped[i]
  }

  return {
    indices: decimatedIndices,
    decimated: true,
  }
}

const buildRangeData = (
  range: Range,
  targetPoints: number,
  dynamicDownsampling: boolean,
  zoomThreshold: number
) => {
  if (!timeData || !rawData) {
    return {
      indices: new Uint32Array(0),
      decimated: false,
      visibleRange: null as Range,
    }
  }

  if (timeData.length === 0) {
    return {
      indices: new Uint32Array(0),
      decimated: false,
      visibleRange: null as Range,
    }
  }

  const firstTime = timeData[0]
  const lastTime = timeData[timeData.length - 1]
  let startIdx = 0
  let endIdx = timeData.length
  let visibleRange: Range = null

  if (range && dynamicDownsampling && lastTime !== firstTime) {
    const span = lastTime - firstTime
    const requestedMin = clampRange(range.min, firstTime, lastTime)
    const requestedMax = clampRange(range.max, firstTime, lastTime)
    const ratio = (requestedMax - requestedMin) / span

    if (ratio <= zoomThreshold) {
      const low = Math.max(0, binarySearch(timeData, requestedMin, false) - 1)
      const high = Math.min(timeData.length, binarySearch(timeData, requestedMax, true) + 1)
      startIdx = Math.max(0, Math.min(low, timeData.length - 1))
      endIdx = Math.max(startIdx + 1, high)
      visibleRange = { min: requestedMin, max: requestedMax }
    }
  }

  if (visibleRange === null && range) {
    visibleRange = {
      min: Math.max(range.min, firstTime),
      max: Math.min(range.max, lastTime),
    }
  }

  if (!visibleRange && timeData.length > 0) {
    visibleRange = { min: firstTime, max: lastTime }
  }

  const { indices, decimated } = decimateSegment(startIdx, endIdx, targetPoints)
  cachedRange = visibleRange
  return { indices, decimated, visibleRange: cachedRange }
}

const respondWithRange = (
  requestId: string | undefined,
  indices: Uint32Array,
  decimated: boolean,
  visibleRange: Range
) => {
  const slices = buildSlicesFromIndices(indices)
  const widthSegments = buildWidthSegmentsForRange(visibleRange)
  postMessage(
    {
      type: 'RANGE_DATA',
      requestId,
      payload: {
        time: slices.time,
        raw: slices.raw,
        filtered: slices.filtered,
        displayedPoints: slices.time.length,
        totalPoints,
        visibleRange,
        decimated,
        widthSegments,
      },
    },
    [
      slices.time.buffer,
      slices.raw.buffer,
      ...(slices.filtered ? [slices.filtered.buffer] : []),
      ...(widthSegments
        ? [widthSegments.x0.buffer, widthSegments.x1.buffer, widthSegments.y.buffer]
        : []),
    ]
  )
}

const handleSetData = (message: SetDataMessage) => {
  timeData = message.payload.time
  if (message.payload.timeUnit === 'seconds' && timeData) {
    for (let i = 0; i < timeData.length; i += 1) {
      timeData[i] = timeData[i] / 60
    }
  }

  rawData = message.payload.raw
  filteredData = message.payload.filtered || null
  totalPoints = message.payload.totalPoints ?? (timeData ? timeData.length : 0)

  let timeRange: Range = null
  if (message.payload.timeRange) {
    timeRange = message.payload.timeRange
  } else if (timeData && timeData.length > 0) {
    timeRange = {
      min: timeData[0],
      max: timeData[timeData.length - 1],
    }
  }

  postMessage({
    type: 'READY',
    payload: {
      totalPoints,
      timeRange,
    },
  })
}

const handleUpdateFiltered = (message: UpdateFilteredMessage) => {
  filteredData = message.payload.filtered
}

const handleSetPeakProperties = (message: SetPeakPropertiesMessage) => {
  peakLeftIps = message.payload.leftIps || null
  peakRightIps = message.payload.rightIps || null
  peakWidthHeights = message.payload.widthHeights || null
}

const handleGetRange = (message: GetRangeMessage) => {
  if (!timeData || !rawData) {
    postMessage({
      type: 'ERROR',
      requestId: message.requestId,
      payload: {
        message: 'Data has not been initialized in worker',
      },
    })
    return
  }

  const { range, targetPoints, dynamicDownsampling, zoomThreshold } = message.payload
  const { indices, decimated, visibleRange } = buildRangeData(
    range,
    targetPoints,
    dynamicDownsampling,
    zoomThreshold
  )
  respondWithRange(message.requestId, indices, decimated, visibleRange)
}

const handleClear = () => {
  timeData = null
  rawData = null
  filteredData = null
  peakLeftIps = null
  peakRightIps = null
  peakWidthHeights = null
  totalPoints = 0
  cachedRange = null
}

ctx.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data
  switch (message.type) {
    case 'SET_DATA':
      handleSetData(message)
      break
    case 'UPDATE_FILTERED':
      handleUpdateFiltered(message)
      break
    case 'SET_PEAK_PROPERTIES':
      handleSetPeakProperties(message)
      break
    case 'GET_RANGE':
      handleGetRange(message)
      break
    case 'CLEAR':
      handleClear()
      break
    default:
      postMessage({
        type: 'ERROR',
        requestId: (message as BaseMessage).requestId,
        payload: { message: 'Unknown worker message received' },
      })
  }
}

export {}
