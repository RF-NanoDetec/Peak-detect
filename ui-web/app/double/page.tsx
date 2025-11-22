"use client"

import { useState, useMemo } from "react"
import { GitBranch, CheckCircle } from "lucide-react"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useRouter } from "next/navigation"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { computePairMetrics, filterPairMetricsByIndices } from "@/components/double/metrics"
import { DoublePeakPanel } from "@/components/double/DoublePeakPanel"
import { DoublePeakScatter } from "@/components/double/DoublePeakScatter"
import { ExportControl } from "@/components/shared/ExportControl"
import type { DoublePeakThresholds } from "@/components/double/types"
import type { PairMetrics } from "@/components/double/metrics"

export default function DoublePeakPage() {
  const router = useRouter()
  const { detectionResults } = useResultsStore()
  const { params } = useParamsStore()

  const [selectedIndices, setSelectedIndices] = useState<number[]>([])
  const [thresholds, setThresholds] = useState<DoublePeakThresholds>({
    distance: [0.1, 100.0] as [number, number],
    pairPromRatio: [0.01, 100.0] as [number, number],
    pairWidthRatio: [0.01, 100.0] as [number, number],
    promOverAmp: [0.0, 1.0] as [number, number],
  })

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

  // Always show full metrics in histograms (not filtered)
  // Filtering is applied in export only
  const displayMetrics = fullMetrics

  const hasData = fullMetrics.totalPairs > 0

  const doublePeakParams = {
    min_distance: thresholds.distance[0] / 1000,
    max_distance: thresholds.distance[1] / 1000,
    min_amp_ratio: thresholds.pairPromRatio[0],
    max_amp_ratio: thresholds.pairPromRatio[1],
    min_width_ratio: thresholds.pairWidthRatio[0],
    max_width_ratio: thresholds.pairWidthRatio[1],
  }

  return (
    <PageShell>
      <PageVisualization>
        {hasData ? (
          <div className="flex-1 p-4 overflow-auto">
            <DoublePeakScatter
              metrics={fullMetrics}
              thresholds={thresholds}
              selectedIndices={selectedIndices}
              onSelectedIndicesChange={setSelectedIndices}
            />
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center space-y-4 max-w-md">
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <GitBranch className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">No Pair Data</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Run peak detection to analyze consecutive peak pairs
                </p>
              </div>
              <Button onClick={() => router.push('/preprocess')}>
                Go to Process & Detect
              </Button>
            </div>
          </div>
        )}
      </PageVisualization>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Double Peak</h2>
            <p className="text-sm text-muted-foreground">
              Analyze pairs of consecutive peaks
            </p>
          </div>

          {hasData ? (
            <>
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <CardTitle className="text-base">Analysis Ready</CardTitle>
                  </div>
                  <CardDescription>Pair metrics computed from detected peaks</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <p className="text-muted-foreground">Total Peaks</p>
                      <p className="text-lg font-semibold">{fullMetrics.totalPeaks}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Total Pairs</p>
                      <p className="text-lg font-semibold">{fullMetrics.totalPairs}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <DoublePeakPanel
                metrics={displayMetrics}
                thresholds={thresholds}
                onThresholdsChange={setThresholds}
              />
              <ExportControl 
                hasDoublePeakAnalysis={true} 
                doublePeakParams={doublePeakParams}
              />
            </>
          ) : (
            <div className="text-center py-8 space-y-3">
              <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                <GitBranch className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">No Peak Data</h3>
                <p className="text-xs text-muted-foreground mt-1">
                  Detect peaks first to analyze pairs
                </p>
              </div>
              <Button size="sm" onClick={() => router.push('/preprocess')}>
                Go to Process & Detect
              </Button>
            </div>
          )}
        </div>
      </PageControls>
    </PageShell>
  )
}
