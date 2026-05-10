"use client"

import { useState, useMemo, useEffect } from "react"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useDoubleStore } from "@/lib/stores/doubleStore"
import { computePairMetrics, filterPairsByThresholds, filterPairMetricsByIndices, normalizeDoubleThresholds } from "@/components/double/metrics"
import { DoublePeakPanel } from "@/components/double/DoublePeakPanel"
import { DoublePeakScatter } from "@/components/double/DoublePeakScatter"
import type { DoublePeakThresholds } from "@/components/double/types"
import type { PairMetrics } from "@/components/double/metrics"
import { advancedWorkflowSteps, workflowSteps } from "@/components/layout/workflow"

const doublePeakStep = advancedWorkflowSteps.find((step) => step.id === "double")!
const preprocessStep = workflowSteps.find((step) => step.id === "preprocess")!

export default function DoublePeakPage() {
  const router = useRouter()
  const { detectionResults } = useResultsStore()
  const { params } = useParamsStore()
  const resolutionMs = (params.time_resolution || 1e-4) * 1000
  const { selectedPairIndices, setSelectedPairIndices, clearSelectedPairIndices } = useDoubleStore()

  const [selectedIndices, setSelectedIndices] = useState<number[]>(selectedPairIndices)
  const [thresholds, setThresholds] = useState<DoublePeakThresholds>(() => {
    const initialMin = Math.max(1, resolutionMs)
    return {
      distance: [initialMin, 100.0],
      pairPromRatio: [0.0, 100.0],
      pairWidthRatio: [0.0, 100.0],
      promOverAmp: [0.0, 200.0], // Stored in percent for UI
    }
  })

  useEffect(() => {
    // Keep min distance aligned with time resolution
    setThresholds((prev) => {
      const currentMin = prev.distance[0]
      const enforcedMin = Math.max(Math.max(resolutionMs, 1), currentMin)
      if (enforcedMin === currentMin) {
        return prev
      }
      const nextMax = Math.max(prev.distance[1], enforcedMin)
      return {
        ...prev,
        distance: [enforcedMin, nextMax],
      }
    })
  }, [resolutionMs])

  // Compute pair metrics from detection results
  const fullMetrics: PairMetrics = useMemo(() => {
    if (!detectionResults || !detectionResults.peak_times || detectionResults.peak_times.length === 0) {
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

    const peakTimes = detectionResults.peak_times || []
    const peakAmplitudes = detectionResults.peak_amplitudes || []
    const prominences = detectionResults.properties?.prominences || []
    const widths = detectionResults.properties?.widths || []
    const timeResolution = params.time_resolution || 1e-4

    if (peakTimes.length === 0 || prominences.length === 0) {
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

    return computePairMetrics({
      peakTimes,
      peakAmplitudes,
      prominences,
      widths,
      timeResolution,
    })
  }, [detectionResults, params.time_resolution])

  const normalizedThresholds = useMemo(
    () => normalizeDoubleThresholds(thresholds),
    [thresholds]
  )

  const filteredIndices = useMemo(() => {
    if (fullMetrics.totalPairs === 0) return []
    const lassoSelection = selectedIndices.length > 0 ? selectedIndices : undefined
    return filterPairsByThresholds(fullMetrics, normalizedThresholds, lassoSelection)
  }, [fullMetrics, normalizedThresholds, selectedIndices])

  const filteredMetrics = useMemo(() => {
    if (filteredIndices.length === 0) {
      return {
        distanceMs: [],
        pairPromRatio: [],
        pairWidthRatio: [],
        timesMin: [],
        promOverAmp: [],
        totalPairs: 0,
        totalPeaks: fullMetrics.totalPeaks,
      }
    }
    return filterPairMetricsByIndices(fullMetrics, filteredIndices)
  }, [fullMetrics, filteredIndices])

  // Keep local selection in sync with store (used by export page)
  useEffect(() => {
    setSelectedIndices(selectedPairIndices)
  }, [selectedPairIndices])

  // Clamp selection when pair count changes
  useEffect(() => {
    setSelectedIndices((prev) => {
      const clamped = prev.filter((idx) => idx >= 0 && idx < fullMetrics.totalPairs)
      if (clamped.length !== prev.length) {
        setSelectedPairIndices(clamped)
      }
      return clamped
    })
  }, [fullMetrics.totalPairs, setSelectedPairIndices])

  // Always show full metrics in histograms (not filtered)
  // Filtering is applied in export only
  const displayMetrics = fullMetrics

  const hasData = fullMetrics.totalPairs > 0

  useEffect(() => {
    if (!hasData) {
      setSelectedIndices([])
      clearSelectedPairIndices()
    }
  }, [hasData, clearSelectedPairIndices])

  return (
    <PageShell>
      <PageVisualization>
        {hasData ? (
          <div className="flex-1 p-4 overflow-auto">
            <DoublePeakScatter
              metrics={fullMetrics}
              thresholds={thresholds}
              selectedIndices={selectedIndices}
              filteredIndices={filteredIndices}
              onSelectedIndicesChange={(indices) => {
                setSelectedIndices(indices)
                setSelectedPairIndices(indices)
              }}
            />
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center space-y-4 max-w-md">
              <div className="mx-auto w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center">
                <doublePeakStep.icon className="h-8 w-8 text-accent" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">No Pair Data</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Run peak detection to analyze consecutive peak pairs
                </p>
              </div>
              <Button onClick={() => router.push(preprocessStep.href)}>
                Go to {preprocessStep.label}
              </Button>
            </div>
          </div>
        )}
      </PageVisualization>
      <PageControls widthClassName="w-[360px] lg:w-[430px] xl:w-[500px]">
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">{doublePeakStep.label}</h2>
            <p className="text-sm text-muted-foreground">
              {doublePeakStep.description}
            </p>
          </div>

          {hasData ? (
            <>
              <DoublePeakPanel
                metrics={displayMetrics}
                thresholds={thresholds}
                onThresholdsChange={setThresholds}
                resolutionMs={resolutionMs}
                filteredIndices={filteredIndices}
                filteredMetrics={filteredMetrics}
              />
            </>
          ) : (
            <div className="text-center py-8 space-y-3">
              <div className="mx-auto w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center">
                <doublePeakStep.icon className="h-6 w-6 text-accent" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">No Peak Data</h3>
                <p className="text-xs text-muted-foreground mt-1">
                  Detect peaks first to analyze pairs
                </p>
              </div>
              <Button size="sm" onClick={() => router.push(preprocessStep.href)}>
                Go to {preprocessStep.label}
              </Button>
            </div>
          )}
        </div>
      </PageControls>
    </PageShell>
  )
}
