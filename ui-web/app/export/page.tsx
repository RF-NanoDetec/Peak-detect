"use client"

import { useState } from "react"
import { Download, Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { apiClient } from "@/lib/apiClient"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useResultsStore } from "@/lib/stores/resultsStore"

export default function ExportPage() {
  const { resultId, filteredResultId } = useDataStore()
  const { params } = useParamsStore()
  const { doublePeakResults } = useResultsStore()
  
  const [loading, setLoading] = useState<string | null>(null)

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  const handleExportPeaks = async () => {
    if (!resultId) {
      toast.error("Please load data first")
      return
    }

    setLoading('peaks')
    try {
      const blob = await apiClient.exportPeaksCSV(
        resultId,
        filteredResultId || undefined,
        {
          prominence_threshold: params.prominence_threshold,
          distance: params.distance,
          rel_height: params.rel_height,
          width_ms: params.width_ms,
          prominence_ratio: params.prominence_ratio,
          time_resolution: params.time_resolution,
        }
      )
      downloadBlob(blob, 'peaks.csv')
      toast.success('Peak information exported successfully')
    } catch (error: any) {
      console.error('Failed to export peaks:', error)
      toast.error(error.response?.data?.detail || 'Failed to export peaks')
    } finally {
      setLoading(null)
    }
  }

  const handleExportDoublePeaks = async () => {
    if (!resultId || !doublePeakResults) {
      toast.error("Please run double peak analysis first")
      return
    }

    setLoading('double')
    try {
      const blob = await apiClient.exportDoublePeaksCSV(resultId, doublePeakResults)
      downloadBlob(blob, 'double_peaks.csv')
      toast.success('Double peak data exported successfully')
    } catch (error: any) {
      console.error('Failed to export double peaks:', error)
      toast.error(error.response?.data?.detail || 'Failed to export double peaks')
    } finally {
      setLoading(null)
    }
  }

  return (
    <PageShell>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Export</h2>
            <p className="text-sm text-muted-foreground">
              Export analysis results
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Data Export</CardTitle>
              <CardDescription>
                Export peak data to CSV
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button 
                className="w-full justify-start" 
                variant="outline"
                onClick={handleExportPeaks}
                disabled={loading !== null || !resultId}
              >
                {loading === 'peaks' ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Download className="h-4 w-4 mr-2" />
                )}
                Export Peak Information
              </Button>
              <Button 
                className="w-full justify-start" 
                variant="outline"
                onClick={handleExportDoublePeaks}
                disabled={loading !== null || !resultId || !doublePeakResults}
              >
                {loading === 'double' ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Download className="h-4 w-4 mr-2" />
                )}
                Export Double Peak Data
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Export Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Data Loaded:</span>
                  <span className={resultId ? "text-green-600" : "text-yellow-600"}>
                    {resultId ? "✓ Yes" : "✗ No"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Filtered Data:</span>
                  <span className={filteredResultId ? "text-green-600" : "text-gray-400"}>
                    {filteredResultId ? "✓ Yes" : "○ Optional"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Double Peak Analysis:</span>
                  <span className={doublePeakResults ? "text-green-600" : "text-gray-400"}>
                    {doublePeakResults ? "✓ Yes" : "○ Optional"}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </PageControls>

      <PageVisualization>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center space-y-4 max-w-md">
            <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
              <Download className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Ready to Export</h3>
              <p className="text-sm text-muted-foreground mt-2">
                Choose export options from the left panel. Peak information and plots can be exported
                in various formats for use in publications and reports.
              </p>
            </div>
          </div>
        </div>
      </PageVisualization>
    </PageShell>
  )
}
