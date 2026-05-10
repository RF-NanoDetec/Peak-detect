import type { Parameters } from "@/lib/types"

export interface ParameterValidationContext {
  signalMax?: number | null
  durationMs?: number | null
}

export interface ParameterValidationResult {
  params: Parameters
  messages: string[]
}

const DEFAULT_TIME_RESOLUTION = 1e-4
const MIN_UI_WIDTH_MS = 0.1
const MIN_CUTOFF_HZ = 0.01

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value)

const roundTo = (value: number, decimals: number) => {
  const scale = 10 ** decimals
  return Math.round(value * scale) / scale
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max)

const formatNumber = (value: number) =>
  Number.isInteger(value) ? value.toString() : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")

const parseWidthMs = (widthMs: string): [number, number] | null => {
  const parts = widthMs.split(",").map((part) => Number.parseFloat(part.trim()))
  if (parts.length !== 2 || parts.some((part) => !Number.isFinite(part))) {
    return null
  }
  return [parts[0], parts[1]]
}

export function getParameterBounds(params: Parameters, context: ParameterValidationContext = {}) {
  const timeResolution = params.time_resolution > 0 ? params.time_resolution : DEFAULT_TIME_RESOLUTION
  const resolutionMs = timeResolution * 1000
  const minWidthMs = roundTo(Math.max(MIN_UI_WIDTH_MS, resolutionMs), 6)
  const maxWidthMs = isFiniteNumber(context.durationMs) && context.durationMs > minWidthMs
    ? roundTo(context.durationMs, 6)
    : null
  const maxProminence = isFiniteNumber(context.signalMax) && context.signalMax > 0
    ? context.signalMax
    : null
  const defaultProminence = maxProminence ? roundTo(maxProminence * 0.1, 3) : 10
  const nyquistHz = 1 / (2 * timeResolution)

  return {
    resolutionMs,
    minWidthMs,
    maxWidthMs,
    minDistanceSamples: 1,
    minDistanceMs: resolutionMs,
    maxProminence,
    defaultProminence: Math.max(defaultProminence, Number.EPSILON),
    minCutoffHz: MIN_CUTOFF_HZ,
    maxCutoffHz: nyquistHz * 0.95,
  }
}

export function sanitizeParameters(
  params: Parameters,
  context: ParameterValidationContext = {},
  scope: "detect" | "filter" | "all" = "all",
): ParameterValidationResult {
  const bounds = getParameterBounds(params, context)
  const next: Parameters = { ...params }
  const messages: string[] = []

  if (scope === "filter" || scope === "all") {
    const cutoff = isFiniteNumber(next.filter_cutoff_freq) ? next.filter_cutoff_freq : 1000
    const clampedCutoff = roundTo(clamp(cutoff, bounds.minCutoffHz, bounds.maxCutoffHz), 3)
    if (cutoff !== clampedCutoff) {
      next.filter_cutoff_freq = clampedCutoff
      messages.push(`Cutoff frequency was adjusted to ${formatNumber(clampedCutoff)} Hz; it must be between ${formatNumber(bounds.minCutoffHz)} Hz and ${formatNumber(bounds.maxCutoffHz)} Hz for the loaded time resolution.`)
    }
  }

  if (scope === "detect" || scope === "all") {
    const prominence = isFiniteNumber(next.prominence_threshold) ? next.prominence_threshold : bounds.defaultProminence
    let correctedProminence = prominence
    if (prominence <= 0) {
      correctedProminence = bounds.defaultProminence
      messages.push(`Prominence was set to ${formatNumber(correctedProminence)} counts; it cannot be zero or negative.`)
    } else if (bounds.maxProminence !== null && prominence > bounds.maxProminence) {
      correctedProminence = bounds.maxProminence
      messages.push(`Prominence was adjusted to ${formatNumber(correctedProminence)} counts; it cannot exceed the highest loaded data point.`)
    }
    next.prominence_threshold = roundTo(correctedProminence, 3)

    const distance = Number.isFinite(next.distance) ? Math.round(next.distance) : bounds.minDistanceSamples
    const clampedDistance = Math.max(bounds.minDistanceSamples, distance)
    if (distance !== clampedDistance) {
      messages.push(`Peak distance was adjusted to ${formatNumber(bounds.minDistanceMs)} ms; it must be at least one sample.`)
    }
    next.distance = clampedDistance

    const width = parseWidthMs(next.width_ms)
    let minWidth = width ? width[0] : bounds.minWidthMs
    let maxWidth = width ? width[1] : Math.max(200, bounds.minWidthMs)
    const originalWidth = `${minWidth},${maxWidth}`

    minWidth = clamp(minWidth, bounds.minWidthMs, bounds.maxWidthMs ?? Number.POSITIVE_INFINITY)
    maxWidth = clamp(maxWidth, bounds.minWidthMs, bounds.maxWidthMs ?? Number.POSITIVE_INFINITY)
    if (maxWidth < minWidth) {
      maxWidth = minWidth
    }

    const correctedWidth = `${formatNumber(roundTo(minWidth, 6))},${formatNumber(roundTo(maxWidth, 6))}`
    if (!width || originalWidth !== `${minWidth},${maxWidth}`) {
      const upper = bounds.maxWidthMs ? ` and below ${formatNumber(bounds.maxWidthMs)} ms` : ""
      messages.push(`Peak width was adjusted to ${correctedWidth} ms; width must be at least the loaded time resolution (${formatNumber(bounds.minWidthMs)} ms)${upper}.`)
    }
    next.width_ms = correctedWidth

    const relHeight = isFiniteNumber(next.rel_height) ? next.rel_height : 0.8
    const clampedRelHeight = clamp(relHeight, Number.EPSILON, 1)
    if (relHeight !== clampedRelHeight) {
      messages.push("Relative height was adjusted; it must be greater than 0% and no more than 100%.")
    }
    next.rel_height = clampedRelHeight

    const prominenceRatio = isFiniteNumber(next.prominence_ratio) ? next.prominence_ratio : 0
    const clampedProminenceRatio = clamp(prominenceRatio, 0, 1)
    if (prominenceRatio !== clampedProminenceRatio) {
      messages.push("Prominence ratio was adjusted; it must be between 0% and 100%.")
    }
    next.prominence_ratio = clampedProminenceRatio
  }

  return { params: next, messages }
}
