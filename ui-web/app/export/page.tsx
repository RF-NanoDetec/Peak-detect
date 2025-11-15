"use client"

import { useState } from "react"
import { Download, Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
  const [imageFormat, setImageFormat] = useState<'png' | 'svg' | 'pdf' | 'jpg'>('png')
  const [imageDpi, setImageDpi] = useState(300)

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

  const handleExportPlot = async () => {
    if (!resultId) {
      toast.error("Please load data first")
      return
    }

    setLoading('plot')
    try {
      const blob = await apiClient.exportPlotImage(
        resultId,
        filteredResultId || undefined,
        imageFormat,
        imageDpi
      )
      downloadBlob(blob, `plot.${imageFormat}`)
      toast.success('Plot exported successfully')
    } catch (error: any) {
      console.error('Failed to export plot:', error)
      toast.error(error.response?.data?.detail || 'Failed to export plot')
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
              Export analysis results and visualizations
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
              <CardTitle className="text-base">Image Export</CardTitle>
              <CardDescription>
                Save plots and visualizations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Format</label>
                <select 
                  className="w-full px-3 py-2 rounded-md border bg-background text-sm"
                  value={imageFormat}
                  onChange={(e) => setImageFormat(e.target.value as any)}
                >
                  <option value="png">PNG</option>
                  <option value="jpg">JPEG</option>
                  <option value="svg">SVG</option>
                  <option value="pdf">PDF</option>
                </select>
              </div>
              {(imageFormat === 'png' || imageFormat === 'jpg') && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">DPI</label>
                  <Input
                    type="number"
                    value={imageDpi}
                    onChange={(e) => setImageDpi(parseInt(e.target.value))}
                    step="50"
                    min="72"
                    max="600"
                  />
                </div>
              )}
              <Button 
                className="w-full"
                onClick={handleExportPlot}
                disabled={loading !== null || !resultId}
              >
                {loading === 'plot' ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Export Current Plot
                  </>
                )}
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

