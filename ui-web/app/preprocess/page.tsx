"use client"

import { useState, useEffect } from "react"
import { toast } from "sonner"
import { useRouter } from "next/navigation"

import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { PreprocessControls } from "@/components/preprocess/PreprocessControls"
import { PreprocessView } from "@/components/preprocess/PreprocessView"

import { apiClient } from "@/lib/apiClient"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useWebSocket } from "@/hooks/use-websocket"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { sanitizeParameters, type ParameterValidationContext } from "@/components/preprocess/paramValidation"
import type { Parameters } from "@/lib/types"

const finiteMax = (values?: number[]) => {
  if (!values || values.length === 0) return null
  let max = Number.NEGATIVE_INFINITY
  for (const value of values) {
    if (Number.isFinite(value) && value > max) {
      max = value
    }
  }
  return Number.isFinite(max) ? max : null
}

export default function PreprocessPage() {
  const router = useRouter()
  const { resultId, filteredResultId, setFilteredResultId, previewData, meta } = useDataStore()
  const { params, updateParam } = useParamsStore()
  const { detectionResults, setDetectionResults } = useResultsStore()
  
  const [loading, setLoading] = useState(false)
  const [taskId, setTaskId] = useState<string | null>(null)
  const { progress, status, result } = useWebSocket(taskId)
  const [detecting, setDetecting] = useState(false)
  
  // Local state for prominence threshold input to allow intermediate typing states
  const [prominenceInput, setProminenceInput] = useState<string>(params.prominence_threshold?.toString() ?? '')
  
  useEffect(() => {
    if (status === 'completed' && result) {
      toast.success("Preprocessing completed")
      
      if (typeof result.resultId === 'string') {
        setFilteredResultId(result.resultId)
      }
      
      setTaskId(null)
      setLoading(false)
    } else if (status === 'failed') {
      toast.error("Preprocessing failed")
      setTaskId(null)
      setLoading(false)
    }
  }, [status, result, setFilteredResultId])

  useEffect(() => {
    setProminenceInput(params.prominence_threshold?.toString() ?? '')
  }, [params.prominence_threshold])

  const validationContext: ParameterValidationContext = {
    signalMax: finiteMax(previewData?.filtered_amplitude ?? previewData?.amplitude),
    durationMs: meta?.time_range
      ? Math.max(0, (meta.time_range[1] - meta.time_range[0]) * 1000)
      : null,
  }

  const applyParamCorrections = (
    scope: "detect" | "filter" | "all" = "all",
    sourceParams: Parameters = params,
  ) => {
    const validation = sanitizeParameters(sourceParams, validationContext, scope)
    const corrected = validation.params

    ;(Object.keys(corrected) as Array<keyof Parameters>).forEach((key) => {
      if (corrected[key] !== sourceParams[key]) {
        updateParam(key, corrected[key] as never)
      }
    })

    if (corrected.prominence_threshold !== sourceParams.prominence_threshold) {
      setProminenceInput(corrected.prominence_threshold.toString())
    }

    validation.messages.forEach((message) => toast.warning(message))
    return corrected
  }

  const handleDetectPeaks = async () => {
    if (!resultId) {
      toast.error("Please load data first")
      router.push('/load')
      return
    }
    
    setDetecting(true)
    try {
      const correctedParams = applyParamCorrections("detect")
      const detectParams = {
        resultId,
        ...(filteredResultId ? { filteredResultId } : {}),
        prominence_threshold: correctedParams.prominence_threshold,
        distance: correctedParams.distance,
        rel_height: correctedParams.rel_height,
        width_ms: correctedParams.width_ms,
        prominence_ratio: correctedParams.prominence_ratio,
        time_resolution: correctedParams.time_resolution,
      }
      const response = await apiClient.detectPeaks(detectParams)
      setDetectionResults(response)
      toast.success(`Detected ${response.count} peaks`)
    } catch (error: any) {
      toast.error(error?.message || "Failed to detect peaks")
    } finally {
      setDetecting(false)
    }
  }

  const handleApplyFilter = async () => {
    if (!resultId) {
      toast.error("Please load data first")
      router.push('/load')
      return
    }

    setLoading(true)
    try {
      const correctedParams = applyParamCorrections("filter")
      const response = await apiClient.runPreprocess({
        resultId,
        params: {
          filter_enabled: correctedParams.filter_enabled,
          filter_type: correctedParams.filter_type,
          filter_cutoff_freq: correctedParams.filter_cutoff_freq,
          butter_order: correctedParams.butter_order,
          savgol_window: correctedParams.savgol_window,
          savgol_polyorder: correctedParams.savgol_polyorder,
        },
      })

      setTaskId(response.taskId)
      toast.info("Processing started...")
    } catch (error: any) {
      console.error('Failed to start preprocessing:', error)
      // Error handling logic here (simplified for brevity)
      const msg = error.response?.data?.detail || error.message || 'Failed to start preprocessing'
      toast.error(msg)
      setLoading(false)
    }
  }

  const handleAutoThreshold = async () => {
    if (!resultId) return
    try {
      const response = await apiClient.autoCalculateThreshold(
        resultId,
        5.0,
        params.filter_type !== 'none' && filteredResultId ? filteredResultId : undefined
      )
      updateParam('prominence_threshold', response.prominence_threshold)
      toast.success(`Threshold set to ${response.prominence_threshold.toFixed(2)}`)
    } catch {
      toast.error("Failed to calculate threshold")
    }
  }

  const handleAutoCutoff = async () => {
    if (!resultId) {
      toast.error("Please load data first")
      return
    }
    try {
      const response = await apiClient.autoCalculateCutoff(resultId)
      const cutoffInt = Math.round(response.cutoff_freq)
      updateParam('filter_cutoff_freq', cutoffInt)
      toast.success(`Cutoff set to ${cutoffInt} Hz`)
    } catch (error) {
      toast.error("Failed to calculate cutoff")
    }
  }

  return (
    <PageShell>
      <PageVisualization>
        <PreprocessView
          resultId={resultId}
          filteredResultId={filteredResultId}
          params={params}
          detectionResults={detectionResults}
          onLoadData={() => router.push('/load')}
        />
      </PageVisualization>
      <PageControls>
        <PreprocessControls
          params={params}
          updateParam={updateParam}
          onApplyFilter={handleApplyFilter}
          onDetectPeaks={handleDetectPeaks}
          onAutoThreshold={handleAutoThreshold}
          onAutoCutoff={handleAutoCutoff}
          loading={loading}
          detecting={detecting}
          progress={progress}
          resultId={resultId}
          prominenceInput={prominenceInput}
          setProminenceInput={setProminenceInput}
          onValidateParams={applyParamCorrections}
        />
      </PageControls>
    </PageShell>
  )
}
