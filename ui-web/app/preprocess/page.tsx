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

export default function PreprocessPage() {
  const router = useRouter()
  const { resultId, previewData, filteredResultId, setFilteredResultId } = useDataStore()
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
      
      if (result.resultId) {
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

  const handleDetectPeaks = async () => {
    if (!resultId) {
      toast.error("Please load data first")
      router.push('/load')
      return
    }
    
    setDetecting(true)
    try {
      const detectParams = {
        resultId,
        ...(filteredResultId ? { filteredResultId } : {}),
        prominence_threshold: params.prominence_threshold,
        distance: params.distance,
        rel_height: params.rel_height,
        width_ms: params.width_ms,
        prominence_ratio: params.prominence_ratio,
        time_resolution: params.time_resolution,
      }
      const response = await apiClient.detectPeaks(detectParams)
      setDetectionResults(response)
      const dataType = filteredResultId ? 'filtered' : 'raw'
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
      const response = await apiClient.runPreprocess({
        resultId,
        params: {
          filter_enabled: params.filter_enabled,
          filter_type: params.filter_type,
          filter_cutoff_freq: params.filter_cutoff_freq,
          butter_order: params.butter_order,
          savgol_window: params.savgol_window,
          savgol_polyorder: params.savgol_polyorder,
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
        />
      </PageControls>

      <PageVisualization>
        <PreprocessView
          resultId={resultId}
          filteredResultId={filteredResultId}
          params={params}
          detectionResults={detectionResults}
          previewData={previewData}
          onLoadData={() => router.push('/load')}
        />
      </PageVisualization>
    </PageShell>
  )
}
