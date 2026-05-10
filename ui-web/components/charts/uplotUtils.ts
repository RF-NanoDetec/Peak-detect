import type { MutableRefObject } from "react"
import uPlot from "uplot"

export const SCALE_DISTRIBUTIONS = {
  LOG: 3,
  ARCSINH: 4,
} as const

export const cleanupUPlotInstance = (instanceRef: MutableRefObject<uPlot | null>) => {
  if (!instanceRef.current) return

  try {
    instanceRef.current.destroy()
  } catch {
    // Ignore cleanup errors from partially initialized chart instances.
  }
  instanceRef.current = null
}

export const formatAxisNumber = (value: number | null | undefined): string => {
  if (value == null || Number.isNaN(value) || !Number.isFinite(value)) return ""
  if (value === 0) return "0"
  const abs = Math.abs(value)
  if (abs < 0.001 || abs > 10000) return value.toExponential(1)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  if (abs >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

export const safeArrayMin = (arr: number[]): number => {
  if (arr.length === 0) return 0
  let min = arr[0]
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i]
  }
  return min
}

export const safeArrayMax = (arr: number[]): number => {
  if (arr.length === 0) return 0
  let max = arr[0]
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i]
  }
  return max
}

export const defaultRange = (scaleType: "linear" | "log") =>
  scaleType === "linear" ? { min: 0, max: 100 } : { min: 0.1, max: 1000 }

export const safeLogAxisSplits = (
  _u: uPlot,
  _axisIdx: number,
  scaleMin: number,
  scaleMax: number,
  _foundIncr: number,
  _foundSpace: number
): number[] => {
  if (!Number.isFinite(scaleMin) || !Number.isFinite(scaleMax) || scaleMin <= 0 || scaleMax <= scaleMin) {
    return [1, 10, 100]
  }

  const splits: number[] = []
  const logMin = Math.floor(Math.log10(scaleMin))
  const logMax = Math.ceil(Math.log10(scaleMax))
  const maxOrders = 6
  const startExp = Math.max(logMin, logMax - maxOrders)
  const endExp = Math.min(logMax, logMin + maxOrders)

  for (let exp = startExp; exp <= endExp; exp++) {
    const val = Math.pow(10, exp)
    if (val >= scaleMin * 0.9 && val <= scaleMax * 1.1) {
      splits.push(val)
    }
  }

  if (splits.length < 2) {
    splits.length = 0
    splits.push(scaleMin, scaleMax)
  }

  return splits
}
