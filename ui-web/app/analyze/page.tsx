"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { AnalyzeControls } from "@/components/analyze/AnalyzeControls"
import { AnalyzeView } from "@/components/analyze/AnalyzeView"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"

export default function AnalyzePage() {
  const router = useRouter()
  const { detectionResults } = useResultsStore()
  const { previewData } = useDataStore()
  const { params } = useParamsStore()

  const hasResults = !!(detectionResults && previewData)
  const [binWidthSeconds, setBinWidthSeconds] = useState<number>(10)
  const [rollingMeanWindow, setRollingMeanWindow] = useState<number>(10)

  return (
    <PageShell>
      <PageVisualization>
        <AnalyzeView
          hasResults={hasResults}
          detectionResults={detectionResults}
          params={params}
          binWidthSeconds={binWidthSeconds}
          rollingMeanWindow={rollingMeanWindow}
          onDetectPeaks={() => router.push('/preprocess')}
        />
      </PageVisualization>
      <PageControls>
        <AnalyzeControls
          hasResults={hasResults}
          binWidthSeconds={binWidthSeconds}
          setBinWidthSeconds={setBinWidthSeconds}
          rollingMeanWindow={rollingMeanWindow}
          setRollingMeanWindow={setRollingMeanWindow}
        />
      </PageControls>
    </PageShell>
  )
}
