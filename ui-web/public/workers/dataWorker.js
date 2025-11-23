/// <reference lib="webworker" />

const ctx = self

let timeData = null
let rawData = null
let filteredData = null
let peakLeftIps = null
let peakRightIps = null
let peakWidthHeights = null
let totalPoints = 0
let cachedRange = null

// Coarse level data (Level 1)
let coarseTimeData = null
let coarseRawData = null
let coarseFilteredData = null
const COARSE_TARGET_POINTS = 50000 // Target size for coarse level

const postMessage = (message, transfer = []) => {
    ctx.postMessage(message, transfer)
}

const clampRange = (value, min, max) => {
    if (value < min) return min
    if (value > max) return max
    return value
}

const binarySearch = (array, value, findUpperBound) => {
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

const createSequentialIndices = (start, end) => {
    const length = Math.max(end - start, 0)
    const indices = new Uint32Array(length)
    for (let i = 0; i < length; i += 1) {
        indices[i] = start + i
    }
    return indices
}

const fractionalIndexToTime = (idx) => {
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

const buildWidthSegmentsForRange = (visibleRange) => {
    if (!peakLeftIps || !peakRightIps || !timeData || timeData.length === 0) {
        return null
    }

    const count = Math.min(peakLeftIps.length, peakRightIps.length)
    if (count === 0) {
        return null
    }

    const x0 = []
    const x1 = []
    const y = []

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

const buildSlicesFromIndices = (indices, sourceTime, sourceRaw, sourceFiltered) => {
    if (!sourceTime || !sourceRaw) {
        return {
            time: new Float64Array(0),
            raw: new Float32Array(0),
            filtered: null,
        }
    }

    const timeSlice = new Float64Array(indices.length)
    const rawSlice = new Float32Array(indices.length)
    const hasFiltered = !!sourceFiltered
    const filteredSlice = hasFiltered ? new Float32Array(indices.length) : null

    for (let i = 0; i < indices.length; i += 1) {
        const idx = indices[i]
        timeSlice[i] = sourceTime[idx]
        rawSlice[i] = sourceRaw[idx]
        if (hasFiltered && filteredSlice && sourceFiltered) {
            filteredSlice[i] = sourceFiltered[idx]
        }
    }

    return {
        time: timeSlice,
        raw: rawSlice,
        filtered: filteredSlice,
    }
}

// Modified to accept source data explicitly
const decimateSegment = (startIdx, endIdx, maxPoints, sourceRaw) => {
    const length = Math.max(endIdx - startIdx, 0)

    if (!sourceRaw || length <= 0) {
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
    const outIndices = []

    for (let bin = 0; bin < nBins; bin += 1) {
        const start = startIdx + bin * binSize
        const end = bin === nBins - 1 ? endIdx : Math.min(endIdx, start + binSize)

        if (end <= start) {
            continue
        }

        let localMinIdx = start
        let localMaxIdx = start
        let minVal = sourceRaw[start]
        let maxVal = sourceRaw[start]

        for (let i = start + 1; i < end; i += 1) {
            const value = sourceRaw[i]
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

    const deduped = []
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
    range,
    targetPoints,
    dynamicDownsampling,
    zoomThreshold
) => {
    if (!timeData || !rawData) {
        return {
            indices: new Uint32Array(0),
            decimated: false,
            visibleRange: null,
            useCoarse: false
        }
    }

    if (timeData.length === 0) {
        return {
            indices: new Uint32Array(0),
            decimated: false,
            visibleRange: null,
            useCoarse: false
        }
    }

    const firstTime = timeData[0]
    const lastTime = timeData[timeData.length - 1]

    // Determine visible range
    let visibleRange = null
    if (range && dynamicDownsampling && lastTime !== firstTime) {
        const requestedMin = clampRange(range.min, firstTime, lastTime)
        const requestedMax = clampRange(range.max, firstTime, lastTime)
        visibleRange = { min: requestedMin, max: requestedMax }
    } else if (range) {
        visibleRange = {
            min: Math.max(range.min, firstTime),
            max: Math.min(range.max, lastTime),
        }
    } else {
        visibleRange = { min: firstTime, max: lastTime }
    }

    // Decision: Use Coarse or Raw?
    // Use coarse if:
    // 1. Coarse data exists
    // 2. We are zoomed out enough (visible range covers a large chunk of data)
    // 3. Or if the number of raw points in the visible range is huge

    let useCoarse = false
    let activeTime = timeData
    let activeRaw = rawData

    if (coarseTimeData && coarseRawData) {
        const span = lastTime - firstTime
        const visibleSpan = visibleRange.max - visibleRange.min
        const ratio = visibleSpan / span

        // If we are seeing more than 10% of the data, use coarse
        // Or if the raw points count would be > 5 * targetPoints
        // But simple ratio check is usually enough for LOD
        if (ratio > 0.05) { // If showing > 5% of data, use coarse
            useCoarse = true
            activeTime = coarseTimeData
            activeRaw = coarseRawData
        }
    }

    // Find indices in the ACTIVE dataset
    let startIdx = 0
    let endIdx = activeTime.length

    if (visibleRange && (visibleRange.min > activeTime[0] || visibleRange.max < activeTime[activeTime.length - 1])) {
        const low = Math.max(0, binarySearch(activeTime, visibleRange.min, false) - 1)
        const high = Math.min(activeTime.length, binarySearch(activeTime, visibleRange.max, true) + 1)
        startIdx = Math.max(0, Math.min(low, activeTime.length - 1))
        endIdx = Math.max(startIdx + 1, high)
    }

    const { indices, decimated } = decimateSegment(startIdx, endIdx, targetPoints, activeRaw)
    cachedRange = visibleRange
    return { indices, decimated, visibleRange: cachedRange, useCoarse }
}

const respondWithRange = (
    requestId,
    indices,
    decimated,
    visibleRange,
    useCoarse
) => {
    // Select source arrays based on decision
    const sourceTime = useCoarse ? coarseTimeData : timeData
    const sourceRaw = useCoarse ? coarseRawData : rawData
    const sourceFiltered = useCoarse ? coarseFilteredData : filteredData

    const slices = buildSlicesFromIndices(indices, sourceTime, sourceRaw, sourceFiltered)
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

const generateCoarseData = () => {
    if (!timeData || !rawData || timeData.length <= COARSE_TARGET_POINTS * 2) {
        coarseTimeData = null
        coarseRawData = null
        coarseFilteredData = null
        return
    }

    console.time("Generate Coarse Data")
    // Use decimateSegment on the FULL range to generate coarse indices
    const { indices } = decimateSegment(0, timeData.length, COARSE_TARGET_POINTS, rawData)

    // Build the coarse arrays
    const slices = buildSlicesFromIndices(indices, timeData, rawData, filteredData)
    coarseTimeData = slices.time
    coarseRawData = slices.raw
    coarseFilteredData = slices.filtered
    console.timeEnd("Generate Coarse Data")
    console.log(`Generated coarse data: ${coarseTimeData.length} points (from ${timeData.length})`)
}

const handleSetData = (message) => {
    timeData = message.payload.time
    if (message.payload.timeUnit === 'seconds' && timeData) {
        for (let i = 0; i < timeData.length; i += 1) {
            timeData[i] = timeData[i] / 60
        }
    }

    rawData = message.payload.raw
    filteredData = message.payload.filtered || null
    totalPoints = message.payload.totalPoints ?? (timeData ? timeData.length : 0)

    // Generate coarse level immediately
    generateCoarseData()

    let timeRange = null
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

const handleUpdateFiltered = (message) => {
    filteredData = message.payload.filtered
    // Regenerate coarse data if filtered data changes
    generateCoarseData()
}

const handleSetPeakProperties = (message) => {
    peakLeftIps = message.payload.leftIps || null
    peakRightIps = message.payload.rightIps || null
    peakWidthHeights = message.payload.widthHeights || null
}

const handleGetRange = (message) => {
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

    console.time(`Worker Process ${message.requestId}`)
    const { range, targetPoints, dynamicDownsampling, zoomThreshold } = message.payload
    const { indices, decimated, visibleRange, useCoarse } = buildRangeData(
        range,
        targetPoints,
        dynamicDownsampling,
        zoomThreshold
    )
    respondWithRange(message.requestId, indices, decimated, visibleRange, useCoarse)
    console.timeEnd(`Worker Process ${message.requestId}`)
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
    coarseTimeData = null
    coarseRawData = null
    coarseFilteredData = null
}

ctx.onmessage = (event) => {
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
                requestId: message.requestId,
                payload: { message: 'Unknown worker message received' },
            })
    }
}

