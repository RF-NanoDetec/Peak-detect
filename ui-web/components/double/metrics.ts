/**
 * Double Peak Metrics Computation
 * 
 * Computes pair-wise and per-peak metrics from detection results.
 */
import type { DoublePeakThresholds } from "./types"

export interface PairMetrics {
  // Arrays of length N-1 (for N peaks)
  distanceMs: number[]
  pairPromRatio: number[]
  pairWidthRatio: number[]
  timesMin: number[] // Time of first peak in pair (minutes)
  
  // Array of length N (for all peaks)
  promOverAmp: number[]
  
  // Metadata
  totalPairs: number
  totalPeaks: number
}

export interface DoublePeakInputs {
  peakTimes: number[] // seconds
  peakAmplitudes: number[]
  prominences: number[]
  widths: number[] // samples
  timeResolution: number // seconds per sample
}

/**
 * Compute all pair and per-peak metrics from detection results.
 */
export function computePairMetrics(inputs: DoublePeakInputs): PairMetrics {
  const { peakTimes, peakAmplitudes, prominences, widths, timeResolution } = inputs
  
  const N = peakTimes.length
  
  if (N === 0) {
    return {
      distanceMs: [],
      pairPromRatio: [],
      pairWidthRatio: [],
      timesMin: [],
      promOverAmp: [],
      totalPairs: 0,
      totalPeaks: 0,
    }
  }
  
  // Convert widths from samples to milliseconds
  const widthsMs = widths.map(w => w * timeResolution * 1000)
  
  // Per-peak metric (all N peaks)
  const promOverAmp = peakAmplitudes.map((amp, i) => {
    if (amp === 0 || !Number.isFinite(amp)) return 0
    return prominences[i] / amp
  })
  
  // Pair metrics (N-1 pairs)
  const distanceMs: number[] = []
  const pairPromRatio: number[] = []
  const pairWidthRatio: number[] = []
  const timesMin: number[] = []
  
  for (let i = 0; i < N - 1; i++) {
    // Distance between peak i and peak i+1 in ms
    const dist = (peakTimes[i + 1] - peakTimes[i]) * 1000
    distanceMs.push(dist)
    
    // Prominence ratio: prom[i+1] / prom[i]
    const promRatio = prominences[i] !== 0 && Number.isFinite(prominences[i])
      ? prominences[i + 1] / prominences[i]
      : 0
    pairPromRatio.push(promRatio)
    
    // Width ratio: width[i+1] / width[i]
    const widthRatio = widthsMs[i] !== 0 && Number.isFinite(widthsMs[i])
      ? widthsMs[i + 1] / widthsMs[i]
      : 0
    pairWidthRatio.push(widthRatio)
    
    // Time of first peak in pair (minutes)
    timesMin.push(peakTimes[i] / 60)
  }
  
  return {
    distanceMs,
    pairPromRatio,
    pairWidthRatio,
    timesMin,
    promOverAmp,
    totalPairs: N - 1,
    totalPeaks: N,
  }
}

/**
 * Filter pair metrics by selection indices (from lasso).
 */
export function filterPairMetricsByIndices(
  metrics: PairMetrics,
  selectedIndices: number[]
): PairMetrics {
  if (selectedIndices.length === 0) {
    return metrics
  }
  
  const indexSet = new Set(selectedIndices)
  
  return {
    distanceMs: metrics.distanceMs.filter((_, i) => indexSet.has(i)),
    pairPromRatio: metrics.pairPromRatio.filter((_, i) => indexSet.has(i)),
    pairWidthRatio: metrics.pairWidthRatio.filter((_, i) => indexSet.has(i)),
    timesMin: metrics.timesMin.filter((_, i) => indexSet.has(i)),
    promOverAmp: metrics.promOverAmp.filter((_, i) => indexSet.has(i) || indexSet.has(i + 1)),
    totalPairs: selectedIndices.length,
    totalPeaks: metrics.totalPeaks,
  }
}

export interface FilterThresholds {
  distance: [number, number]
  pairPromRatio: [number, number]
  pairWidthRatio: [number, number]
  promOverAmp: [number, number]
}

export const normalizeDoubleThresholds = (
  thresholds: DoublePeakThresholds
): FilterThresholds => ({
  distance: thresholds.distance,
  pairPromRatio: thresholds.pairPromRatio,
  pairWidthRatio: thresholds.pairWidthRatio,
  promOverAmp: [
    thresholds.promOverAmp[0] / 100,
    thresholds.promOverAmp[1] / 100,
  ],
})

/**
 * Filter pairs by threshold values and optionally lasso selection.
 * Returns array of passing pair indices.
 */
export function filterPairsByThresholds(
  metrics: PairMetrics,
  thresholds: FilterThresholds,
  lassoIndices?: number[]
): number[] {
  const passingIndices: number[] = []
  
  // If lasso selection exists, start with those indices, otherwise check all
  const indicesToCheck = lassoIndices && lassoIndices.length > 0
    ? lassoIndices
    : Array.from({ length: metrics.totalPairs }, (_, i) => i)
  
  for (const i of indicesToCheck) {
    // Check distance threshold
    const dist = metrics.distanceMs[i]
    if (dist < thresholds.distance[0] || dist > thresholds.distance[1]) continue
    
    // Check pair prominence ratio threshold
    const promRatio = metrics.pairPromRatio[i]
    if (promRatio < thresholds.pairPromRatio[0] || promRatio > thresholds.pairPromRatio[1]) continue
    
    // Check pair width ratio threshold
    const widthRatio = metrics.pairWidthRatio[i]
    if (widthRatio < thresholds.pairWidthRatio[0] || widthRatio > thresholds.pairWidthRatio[1]) continue
    
    // Check per-peak prominence/amplitude ratio for both peaks in the pair
    const promAmp1 = metrics.promOverAmp[i]
    const promAmp2 = metrics.promOverAmp[i + 1]
    if (promAmp1 < thresholds.promOverAmp[0] || promAmp1 > thresholds.promOverAmp[1]) continue
    if (promAmp2 < thresholds.promOverAmp[0] || promAmp2 > thresholds.promOverAmp[1]) continue
    
    passingIndices.push(i)
  }
  
  return passingIndices
}

/**
 * Export pair metrics to CSV format.
 */
export function exportPairMetricsToCSV(
  metrics: PairMetrics,
  selectedIndices?: number[]
): string {
  const indices = selectedIndices && selectedIndices.length > 0
    ? selectedIndices
    : Array.from({ length: metrics.totalPairs }, (_, i) => i)
  
  const header = 'pair_index,time_min,distance_ms,pair_prom_ratio,pair_width_ratio,prom_over_amp_peak1,prom_over_amp_peak2\n'
  
  const rows = indices.map(i => {
    const timeMin = metrics.timesMin[i]?.toFixed(4) ?? ''
    const distMs = metrics.distanceMs[i]?.toFixed(3) ?? ''
    const promRatio = metrics.pairPromRatio[i]?.toFixed(4) ?? ''
    const widthRatio = metrics.pairWidthRatio[i]?.toFixed(4) ?? ''
    const promAmp1 = metrics.promOverAmp[i]?.toFixed(4) ?? ''
    const promAmp2 = metrics.promOverAmp[i + 1]?.toFixed(4) ?? ''
    
    return `${i},${timeMin},${distMs},${promRatio},${widthRatio},${promAmp1},${promAmp2}`
  }).join('\n')
  
  return header + rows
}

/**
 * Trigger browser download of CSV file.
 */
export function downloadCSV(csvContent: string, filename: string = 'double_peak_pairs.csv') {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)
  
  link.setAttribute('href', url)
  link.setAttribute('download', filename)
  link.style.visibility = 'hidden'
  
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  URL.revokeObjectURL(url)
}
