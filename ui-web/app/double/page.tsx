"use client"

import { useState } from "react"
import { GitBranch, Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { toast } from "sonner"
import { apiClient } from "@/lib/apiClient"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { useRouter } from "next/navigation"
import type { DoublePeakResponse } from "@/lib/types"

export default function DoublePeakPage() {
  const router = useRouter()
  const { resultId, filteredResultId } = useDataStore()
  const { params } = useParamsStore()
  const { setDoublePeakResults } = useResultsStore()
  
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<DoublePeakResponse | null>(null)
  
  // Double peak constraints (in seconds for backend, show in ms for UI)
  const [minDistance, setMinDistance] = useState(1.0) // ms
  const [maxDistance, setMaxDistance] = useState(100.0) // ms
  const [minAmpRatio, setMinAmpRatio] = useState(0.1)
  const [maxAmpRatio, setMaxAmpRatio] = useState(10.0)
  const [minWidthRatio, setMinWidthRatio] = useState(0.1)
  const [maxWidthRatio, setMaxWidthRatio] = useState(10.0)

  const handleAnalyze = async () => {
    if (!resultId || !filteredResultId) {
      toast.error("Please load and preprocess data, then detect peaks first")
      router.push('/load')
      return
    }

    setLoading(true)
    try {
      const response = await apiClient.analyzeDoublePeaks({
        resultId,
        filteredResultId,
        prominence_threshold: params.prominence_threshold,
        distance: params.distance,
        rel_height: params.rel_height,
        width_ms: params.width_ms,
        prominence_ratio: params.prominence_ratio,
        time_resolution: params.time_resolution,
        min_distance: minDistance / 1000, // Convert ms to seconds
        max_distance: maxDistance / 1000, // Convert ms to seconds
        min_amp_ratio: minAmpRatio,
        max_amp_ratio: maxAmpRatio,
        min_width_ratio: minWidthRatio,
        max_width_ratio: maxWidthRatio,
      })

      setResults(response)
      setDoublePeakResults(response)
      toast.success(`Found ${response.double_peak_count} double peaks out of ${response.total_pairs} pairs`)
    } catch (error: any) {
      console.error('Failed to analyze double peaks:', error)
      
      let errorMessage = 'Failed to analyze double peaks'
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
    } finally {
      setLoading(false)
    }
  }

  // Filter to only show double peak pairs
  const doublePeaks = results?.peak_pairs.filter(p => p.is_double_peak) || []

  return (
    <PageShell>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Double Peak</h2>
            <p className="text-sm text-muted-foreground">
              Detect and analyze double peak patterns
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Distance Constraints</CardTitle>
              <CardDescription>
                Set min/max distance between peaks (milliseconds)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Min Distance (ms)</label>
                <Input
                  type="number"
                  value={minDistance}
                  onChange={(e) => setMinDistance(parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Distance (ms)</label>
                <Input
                  type="number"
                  value={maxDistance}
                  onChange={(e) => setMaxDistance(parseFloat(e.target.value))}
                  step="1"
                  min="0"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Amplitude Ratio</CardTitle>
              <CardDescription>
                Ratio of secondary to primary peak amplitude
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Min Ratio</label>
                <Input
                  type="number"
                  value={minAmpRatio}
                  onChange={(e) => setMinAmpRatio(parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Ratio</label>
                <Input
                  type="number"
                  value={maxAmpRatio}
                  onChange={(e) => setMaxAmpRatio(parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Width Ratio</CardTitle>
              <CardDescription>
                Ratio of secondary to primary peak width
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Min Ratio</label>
                <Input
                  type="number"
                  value={minWidthRatio}
                  onChange={(e) => setMinWidthRatio(parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Ratio</label>
                <Input
                  type="number"
                  value={maxWidthRatio}
                  onChange={(e) => setMaxWidthRatio(parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                />
              </div>
            </CardContent>
          </Card>

          <Button 
            className="w-full" 
            onClick={handleAnalyze}
            disabled={loading || !resultId || !filteredResultId}
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze Double Peaks'
            )}
          </Button>

          {results && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Pairs:</span>
                    <span className="font-medium">{results.total_pairs}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Double Peaks:</span>
                    <span className="font-medium">{results.double_peak_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Percentage:</span>
                    <span className="font-medium">
                      {((results.double_peak_count / results.total_pairs) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </PageControls>

      <PageVisualization>
        {results && doublePeaks.length > 0 ? (
          <div className="flex-1 p-4 overflow-auto">
            <h3 className="text-lg font-semibold mb-4">Double Peak Pairs</h3>
            <div className="space-y-3">
              {doublePeaks.map((pair, idx) => (
                <Card key={idx}>
                  <CardHeader className="py-3">
                    <CardTitle className="text-sm">Pair #{idx + 1}</CardTitle>
                  </CardHeader>
                  <CardContent className="py-2">
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-muted-foreground">Distance:</span>
                        <span className="ml-2 font-medium">{pair.peak_distance_ms.toFixed(2)} ms</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Amp Ratio:</span>
                        <span className="ml-2 font-medium">{pair.amplitude_ratio.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Width Ratio:</span>
                        <span className="ml-2 font-medium">{pair.width_ratio.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Start Distance:</span>
                        <span className="ml-2 font-medium">{pair.start_distance_ms.toFixed(2)} ms</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center space-y-4 max-w-md">
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <GitBranch className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">
                  {results ? 'No Double Peaks Found' : 'No Analysis Yet'}
                </h3>
                <p className="text-sm text-muted-foreground mt-2">
                  {results 
                    ? 'Adjust the constraints to find double peak patterns'
                    : 'Run peak detection first, then analyze double peak patterns'
                  }
                </p>
              </div>
              {!results && (
                <Button onClick={() => router.push('/detect')}>
                  Detect Peaks
                </Button>
              )}
            </div>
          </div>
        )}
      </PageVisualization>
    </PageShell>
  )
}

