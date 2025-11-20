"use client"

import { useState, useEffect } from "react"
import { Filter, Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { PreprocessingChart } from "@/components/charts/PreprocessingChart"
import { toast } from "sonner"
import { apiClient } from "@/lib/apiClient"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useWebSocket } from "@/hooks/use-websocket"
import { useRouter } from "next/navigation"
import { useResultsStore } from "@/lib/stores/resultsStore"

export default function PreprocessPage() {
  const router = useRouter()
  const { resultId, previewData, filteredResultId, setFilteredResultId, meta } = useDataStore()
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
      
      // Store the filtered result ID in global state
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

  // Sync prominence input with params when it changes externally (e.g., from Auto Threshold button)
  useEffect(() => {
    setProminenceInput(params.prominence_threshold?.toString() ?? '')
  }, [params.prominence_threshold])

  const handleDetectPeaks = async () => {
    if (!resultId) {
      toast.error("Please load data first")
      router.push('/load')
      return
    }
    
    // Log what we're detecting on
    console.log('Detecting peaks:', { 
      resultId, 
      filteredResultId, 
      useFiltered: !!filteredResultId,
      threshold: params.prominence_threshold,
      distance: params.distance,
      allParams: params
    })
    
    setDetecting(true)
    try {
      const detectParams = {
        resultId,
        // If filtered result exists, always use it for detection
        ...(filteredResultId ? { filteredResultId } : {}),
        prominence_threshold: params.prominence_threshold,
        distance: params.distance,
        rel_height: params.rel_height,
        width_ms: params.width_ms,
        prominence_ratio: params.prominence_ratio,
        time_resolution: params.time_resolution,
      }
      console.log('Sending detect request with params:', detectParams)
      const response = await apiClient.detectPeaks(detectParams)
      setDetectionResults(response)
      const dataType = filteredResultId ? 'filtered' : 'raw'
      console.log(`Detected ${response.count} peaks on ${dataType} data`)
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
      
      // Handle different error formats
      let errorMessage = 'Failed to start preprocessing'
      if (error.response?.data) {
        const data = error.response.data
        if (typeof data.detail === 'string') {
          errorMessage = data.detail
        } else if (Array.isArray(data.detail)) {
          errorMessage = data.detail.map((err: any) => 
            `${err.loc?.join('.')}: ${err.msg}`
          ).join(', ')
        } else if (data.message) {
          errorMessage = data.message
        }
      } else if (error.message) {
        errorMessage = error.message
      }
      
      toast.error(errorMessage)
      setLoading(false)
    }
  }

  return (
    <PageShell>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Process & Detect</h2>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Filter Type</CardTitle>
              <CardDescription className="text-xs">
                Choose a filter to apply to the signal
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-xs font-medium">Filter</label>
                <select 
                  className="w-full px-3 py-2 rounded-md border bg-background text-xs"
                  value={params.filter_type}
                  onChange={(e) => updateParam('filter_type', e.target.value as any)}
                >
                  <option value="none">None</option>
                  <option value="butterworth">Butterworth</option>
                  <option value="savgol">Savitzky-Golay</option>
                </select>
              </div>
            </CardContent>
          </Card>

          

          {params.filter_type === 'butterworth' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Butterworth Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <label className="text-xs font-medium">Cutoff Frequency (Hz)</label>
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      step="1"
                      value={params.filter_cutoff_freq}
                      onChange={(e) => updateParam('filter_cutoff_freq', Math.round(parseFloat(e.target.value) || 0))}
                      className="text-xs"
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
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
                      }}
                      disabled={!resultId}
                    >
                      Auto
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium">Order</label>
                  <Input
                    type="number"
                    value={params.butter_order}
                    onChange={(e) => updateParam('butter_order', parseInt(e.target.value))}
                    className="text-xs"
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {params.filter_type === 'savgol' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Savitzky-Golay Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <label className="text-xs font-medium">Window Length</label>
                  <Input
                    type="number"
                    value={params.savgol_window}
                    onChange={(e) => updateParam('savgol_window', parseInt(e.target.value))}
                    className="text-xs"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium">Polynomial Order</label>
                  <Input
                    type="number"
                    value={params.savgol_polyorder}
                    onChange={(e) => updateParam('savgol_polyorder', parseInt(e.target.value))}
                    className="text-xs"
                  />
                </div>
              </CardContent>
            </Card>
          )}

          <Button 
            className="w-full" 
            onClick={handleApplyFilter}
            disabled={loading || !resultId}
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Processing... {progress}%
              </>
            ) : (
              'Apply Filter'
            )}
          </Button>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Detect Peaks</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">
                    Prominence Threshold
                  </label>
                  <Input
                    type="number"
                    value={prominenceInput}
                    onChange={(e) => {
                      const inputValue = e.target.value
                      // Update local state immediately to show what user is typing
                      setProminenceInput(inputValue)
                      
                      // Update store if value is valid
                      if (inputValue !== '' && inputValue !== '-') {
                        const value = parseFloat(inputValue)
                        if (!isNaN(value) && value >= 0 && isFinite(value)) {
                          updateParam('prominence_threshold', value)
                          console.log('Threshold updated to:', value)
                        }
                      }
                    }}
                    onBlur={(e) => {
                      // Ensure valid value on blur
                      const value = parseFloat(e.target.value)
                      if (isNaN(value) || value < 0 || !isFinite(value)) {
                        const defaultValue = 20
                        setProminenceInput(defaultValue.toString())
                        updateParam('prominence_threshold', defaultValue) // Reset to default
                      } else {
                        // Ensure input shows the parsed value (removes trailing decimals, etc.)
                        setProminenceInput(value.toString())
                      }
                    }}
                    step="0.1"
                    min="0"
                    className="text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Min Distance (samples)</label>
                  <Input
                    type="number"
                    value={params.distance}
                    onChange={(e) => updateParam('distance', parseInt(e.target.value))}
                    className="text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Width Min (ms)</label>
                  <Input
                    type="number"
                    value={params.width_ms.split(',')[0]}
                    onChange={(e) => {
                      const [, max] = params.width_ms.split(',')
                      updateParam('width_ms', `${e.target.value},${max || 200}`)
                    }}
                    step="0.1"
                    min="0.01"
                    className="text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Width Max (ms)</label>
                  <Input
                    type="number"
                    value={params.width_ms.split(',')[1]}
                    onChange={(e) => {
                      const [min] = params.width_ms.split(',')
                      updateParam('width_ms', `${min || 0.1},${e.target.value}`)
                    }}
                    step="1"
                    min="0.1"
                    className="text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Rel Height</label>
                  <div className="relative">
                    <Input
                      type="number"
                      value={Number((params.rel_height * 100).toFixed(1))}
                      onChange={(e) => updateParam('rel_height', parseFloat(e.target.value) / 100)}
                      step="0.5"
                      min="0"
                      max="100"
                      className="pr-10 text-xs"
                    />
                    <span className="absolute inset-y-0 right-3 flex items-center text-[10px] text-muted-foreground">%</span>
                  </div>
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Prominence Ratio</label>
                  <div className="relative">
                    <Input
                      type="number"
                      value={Number((params.prominence_ratio * 100).toFixed(1))}
                      onChange={(e) => updateParam('prominence_ratio', parseFloat(e.target.value) / 100)}
                      step="0.5"
                      min="0"
                      max="100"
                      className="pr-10 text-xs"
                    />
                    <span className="absolute inset-y-0 right-3 flex items-center text-[10px] text-muted-foreground">%</span>
                  </div>
                </div>
              </div>
              <div className="flex gap-2">
                <Button 
                  className="flex-1"
                  onClick={handleDetectPeaks}
                  disabled={detecting || !resultId}
                >
                  {detecting ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Detecting...
                    </>
                  ) : (
                    'Detect Peaks'
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={async () => {
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
                  }}
                  disabled={!resultId}
                >
                  Auto Threshold
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </PageControls>

      <PageVisualization>
        {resultId ? (
          <PreprocessingChart
            resultId={resultId}
            filteredResultId={filteredResultId}
            className="flex-1"
            timeResolution={params.time_resolution}
            peakTimes={detectionResults?.peak_times || []}
            peakAmplitudes={detectionResults?.peak_amplitudes || []}
            peakIntervals={detectionResults?.peak_intervals || []}
            peakProperties={detectionResults?.properties || null}
            previewData={previewData}
            prominenceThreshold={params.prominence_threshold}
            distance={params.distance}
            widthMs={params.width_ms}
            initialHistograms={detectionResults?.histograms}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center space-y-4 max-w-md">
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <Filter className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">No Data to Preprocess</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Load data first to apply preprocessing filters
                </p>
              </div>
              <Button onClick={() => router.push('/load')}>
                Load Data
              </Button>
            </div>
          </div>
        )}
      </PageVisualization>
    </PageShell>
  )
}
