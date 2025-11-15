"use client"

import { useState, useRef, useEffect } from "react"
import { FileUp, FolderOpen, Clock, Loader2, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { toast } from "sonner"
import { apiClient } from "@/lib/apiClient"
import { useDataStore } from "@/lib/stores/dataStore"
import { useParamsStore } from "@/lib/stores/paramsStore"
import { useProtocolStore } from "@/lib/stores/protocolStore"
import { useRouter } from "next/navigation"
import { parseFilePreview, type LocalPreviewResult } from "@/lib/localDataParser"

export default function LoadDataPage() {
  const router = useRouter()
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [localPreview, setLocalPreview] = useState<LocalPreviewResult | null>(null)
  const previewTask = useRef(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const { setResultId, setFiles, setMeta, setPreviewData, setOriginalPreviewData, setFilteredPreviewData, recentFiles, addRecentFiles } = useDataStore()
  const { params, updateParam } = useParamsStore()
  const { photonCorrection, protocol, setApplyCorrection, setDeadTimeNs, updateProtocol } = useProtocolStore()
  
  // Time resolution conversion: display in milliseconds, store in seconds
  const timeResolutionMs = params.time_resolution * 1000
  const handleTimeResolutionChange = (ms: number) => {
    const seconds = ms / 1000
    updateParam('time_resolution', seconds)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      setSelectedFiles(files)
      toast.info(`Selected ${files.length} file(s)`)
    }
  }

  useEffect(() => {
    if (selectedFiles.length === 0) {
      setLocalPreview(null)
      setPreviewError(null)
      setPreviewLoading(false)
      previewTask.current += 1
      return
    }

    const file = selectedFiles[0]
    const taskId = ++previewTask.current
    setPreviewLoading(true)
    setPreviewError(null)

    parseFilePreview(file)
      .then((result) => {
        if (previewTask.current !== taskId) return
        setLocalPreview(result)
        setPreviewError(null)
      })
      .catch((error: any) => {
        if (previewTask.current !== taskId) return
        console.error('Local preview parsing failed:', error)
        setLocalPreview(null)
        setPreviewError(error?.message || 'Failed to parse selected file')
      })
      .finally(() => {
        if (previewTask.current !== taskId) return
        setPreviewLoading(false)
      })
  }, [selectedFiles])

  const handleLoadFiles = async () => {
    if (selectedFiles.length === 0) {
      toast.error("Please select files to load")
      return
    }

    setLoading(true)
    try {
      // Upload files to backend with correction and protocol options
      const response = await apiClient.uploadFiles(selectedFiles, params.time_resolution, {
        applyDeadTimeCorrection: photonCorrection.applyCorrection,
        deadTimeNs: photonCorrection.deadTimeNs,
        protocol: protocol,
      })

      setResultId(response.resultId)
      setFiles(response.meta.files)
      setMeta(response.meta)
      
      // Add to recent files (as a group)
      addRecentFiles(selectedFiles.map(f => f.name))

      // Fetch preview data
      const previewData = await apiClient.getDataPreview(response.resultId)
      setPreviewData(previewData)
      setOriginalPreviewData(previewData)
      setFilteredPreviewData(null) // Clear any previous filtered data

      toast.success(`Loaded ${response.meta.total_files} file(s) successfully`)
      
      // Navigate to preprocess after short delay
      setTimeout(() => router.push('/preprocess'), 500)
    } catch (error: any) {
      console.error('Failed to load files:', error)
      
      // Handle different error formats
      let errorMessage = 'Failed to load files'
      if (error.response?.data) {
        const data = error.response.data
        if (typeof data.detail === 'string') {
          errorMessage = data.detail
        } else if (Array.isArray(data.detail)) {
          // FastAPI validation errors
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

  const handleRecentFilesClick = (fileGroup: string[]) => {
    // Show which files were in this session in the selected files area
    // Note: We display file info but user still needs to actually select files to load
    // This gives better visibility of what files should be selected
    toast.info(`Session files: ${fileGroup.join(', ')}. Please select these files to load.`)
    
    // We'll show the file names in the UI even though they're not actually selected yet
    // This is just for display purposes - actual file selection still required
    // For now, just open the file picker
    fileInputRef.current?.click()
  }

  const formatRecentFileDisplay = (fileGroup: string[]) => {
    if (fileGroup.length === 1) {
      return fileGroup[0]
    }
    return `${fileGroup[0]} + ${fileGroup.length - 1}`
  }

  return (
    <PageShell>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Load Data</h2>
            <p className="text-sm text-muted-foreground">
              Select files to begin analysis
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">File Selection</CardTitle>
              <CardDescription>
                Choose files from your computer
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.xls,.xlsx"
                onChange={handleFileSelect}
                className="hidden"
              />
              <Button 
                className="w-full" 
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
              >
                <Upload className="h-4 w-4 mr-2" />
                Select Files
              </Button>
              {selectedFiles.length > 0 && (
                <div className="text-sm space-y-1">
                  <p className="font-medium">{selectedFiles.length} file(s) selected:</p>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {selectedFiles.map((file, idx) => (
                      <div key={idx} className="text-xs text-muted-foreground truncate" title={file.name}>
                        • {file.name}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Button 
            className="w-full" 
            onClick={handleLoadFiles}
            disabled={loading || selectedFiles.length === 0}
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Loading...
              </>
            ) : (
              <>
                <FileUp className="h-4 w-4 mr-2" />
                Load Files
              </>
            )}
          </Button>

          {recentFiles.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Recent Sessions
                </CardTitle>
                <CardDescription>
                  Previously loaded file groups
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-1">
                  {recentFiles
                    .filter((fileGroup) => Array.isArray(fileGroup) && fileGroup.length > 0)
                    .map((fileGroup, idx) => (
                      <button
                        key={idx}
                        className="w-full text-left px-3 py-2 rounded-md hover:bg-accent text-sm truncate"
                        onClick={() => handleRecentFilesClick(fileGroup)}
                        title={fileGroup.join(', ')}
                      >
                        {formatRecentFileDisplay(fileGroup)}
                      </button>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-base">File Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Time Resolution (milliseconds)</label>
                <Input
                  type="number"
                  placeholder="0.1"
                  value={timeResolutionMs}
                  onChange={(e) => handleTimeResolutionChange(parseFloat(e.target.value))}
                  step="0.01"
                  min="0.001"
                />
                <p className="text-xs text-muted-foreground">
                  Time resolution in milliseconds per sample (default: 0.1 ms)
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Photon Counter Correction</CardTitle>
              <CardDescription>
                Correct for detector non-linearity due to dead time
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-medium">Apply dead-time correction</label>
                  <p className="text-xs text-muted-foreground">
                    Compensates for photon counter blind time
                  </p>
                </div>
                <Switch
                  checked={photonCorrection.applyCorrection}
                  onCheckedChange={setApplyCorrection}
                />
              </div>
              {photonCorrection.applyCorrection && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">Dead time (nanoseconds)</label>
                  <Input
                    type="number"
                    value={photonCorrection.deadTimeNs}
                    onChange={(e) => setDeadTimeNs(parseFloat(e.target.value) || 43.0)}
                    step="1"
                    min="1"
                  />
                  <p className="text-xs text-muted-foreground">
                    Detector dead time T_D (default: 43 ns). Correction uses 1/(1 - R×T_D).
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Protocol Information</CardTitle>
              <CardDescription>
                Experiment metadata for documentation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">Measurement Date</label>
                  <Input
                    type="date"
                    value={protocol.measurement_date || ''}
                    onChange={(e) => updateProtocol('measurement_date', e.target.value)}
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Start Time</label>
                  <Input
                    type="time"
                    value={protocol.start_time || ''}
                    onChange={(e) => updateProtocol('start_time', e.target.value)}
                    className="text-sm"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Setup</label>
                <Input
                  placeholder="e.g., Prototype, Old Ladom"
                  value={protocol.setup || ''}
                  onChange={(e) => updateProtocol('setup', e.target.value)}
                  className="text-sm"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">Sample Number</label>
                  <Input
                    placeholder="Sample ID"
                    value={protocol.sample_number || ''}
                    onChange={(e) => updateProtocol('sample_number', e.target.value)}
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Particle</label>
                  <Input
                    placeholder="Particle type"
                    value={protocol.particle || ''}
                    onChange={(e) => updateProtocol('particle', e.target.value)}
                    className="text-sm"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Concentration</label>
                <Input
                  placeholder="Particle concentration"
                  value={protocol.concentration || ''}
                  onChange={(e) => updateProtocol('concentration', e.target.value)}
                  className="text-sm"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">Buffer</label>
                  <Input
                    placeholder="Buffer solution"
                    value={protocol.buffer || ''}
                    onChange={(e) => updateProtocol('buffer', e.target.value)}
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Buffer Conc.</label>
                  <Input
                    placeholder="Buffer concentration"
                    value={protocol.buffer_concentration || ''}
                    onChange={(e) => updateProtocol('buffer_concentration', e.target.value)}
                    className="text-sm"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">ND Filter</label>
                  <Input
                    placeholder="Filter value"
                    value={protocol.nd_filter || ''}
                    onChange={(e) => updateProtocol('nd_filter', e.target.value)}
                    className="text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium">Laser Power</label>
                  <Input
                    placeholder="Power setting"
                    value={protocol.laser_power || ''}
                    onChange={(e) => updateProtocol('laser_power', e.target.value)}
                    className="text-sm"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Stamp</label>
                <Input
                  placeholder="e.g., triple-block"
                  value={protocol.stamp || ''}
                  onChange={(e) => updateProtocol('stamp', e.target.value)}
                  className="text-sm"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Notes</label>
                <Input
                  placeholder="Additional observations"
                  value={protocol.notes || ''}
                  onChange={(e) => updateProtocol('notes', e.target.value)}
                  className="text-sm"
                />
              </div>
            </CardContent>
          </Card>
        </div>
      </PageControls>

      <PageVisualization>
        {selectedFiles.length === 0 ? (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center space-y-4 max-w-md">
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <FileUp className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">No Data Loaded</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Select files using the file picker to begin your analysis.
                  Supported formats: .txt, .xls, .xlsx
                </p>
              </div>
              <Button onClick={() => fileInputRef.current?.click()}>
                Get Started
              </Button>
            </div>
          </div>
        ) : (
          <LocalPreviewPanel
            loading={previewLoading}
            error={previewError}
            preview={localPreview}
            fileCount={selectedFiles.length}
            onSelectFiles={() => fileInputRef.current?.click()}
          />
        )}
      </PageVisualization>
    </PageShell>
  )
}

interface LocalPreviewPanelProps {
  loading: boolean
  error: string | null
  preview: LocalPreviewResult | null
  fileCount: number
  onSelectFiles: () => void
}

function LocalPreviewPanel({ loading, error, preview, fileCount, onSelectFiles }: LocalPreviewPanelProps) {
  const header =
    preview && preview.columns.length > 0
      ? preview.columns
      : preview?.sampleRows[0]?.map((_, idx) => `Column ${idx + 1}`) ?? []

  return (
    <div className="flex-1 p-6 overflow-auto">
      <div className="max-w-4xl mx-auto space-y-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold">Local Data Preview</h3>
            <p className="text-sm text-muted-foreground">
              Client-side parsing powered by uDSV. Showing first file ({fileCount} selected).
            </p>
          </div>
          <Button variant="outline" onClick={onSelectFiles} size="sm">
            <FolderOpen className="h-4 w-4 mr-2" />
            Choose Different Files
          </Button>
        </div>

        <div className="rounded-2xl border bg-card/80 shadow-sm p-6 min-h-[320px]">
          {loading && (
            <div className="flex flex-col items-center justify-center h-full text-sm text-muted-foreground gap-2">
              <Loader2 className="h-6 w-6 animate-spin" />
              Parsing preview with uDSV...
            </div>
          )}

          {!loading && error && (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-3">
              <p className="text-base font-semibold">Preview failed</p>
              <p className="text-sm text-muted-foreground">{error}</p>
              <Button variant="outline" size="sm" onClick={onSelectFiles}>
                Try Selecting Files Again
              </Button>
            </div>
          )}

          {!loading && !error && preview && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold">{preview.fileName}</p>
                  <p className="text-xs text-muted-foreground">
                    Parsed {preview.rowsParsed.toLocaleString()} rows in {preview.durationMs.toFixed(1)} ms
                    {preview.truncated && ' (preview limited for speed)'}
                  </p>
                </div>
                <div className="text-xs text-muted-foreground">
                  Powered by <a href="https://github.com/leeoniya/uDSV" target="_blank" rel="noreferrer" className="underline">uDSV</a>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                <div className="rounded-lg border bg-background/40 p-3">
                  <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Points Parsed</p>
                  <p className="text-lg font-semibold">{preview.rowsParsed.toLocaleString()}</p>
                </div>
                <div className="rounded-lg border bg-background/40 p-3">
                  <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Time Range</p>
                  <p className="text-lg font-semibold">
                    {formatRange(preview.stats.minTime, preview.stats.maxTime, 's')}
                  </p>
                </div>
                <div className="rounded-lg border bg-background/40 p-3">
                  <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Amplitude Range</p>
                  <p className="text-lg font-semibold">
                    {formatRange(preview.stats.minAmplitude, preview.stats.maxAmplitude)}
                  </p>
                </div>
                <div className="rounded-lg border bg-background/40 p-3">
                  <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Parse Duration</p>
                  <p className="text-lg font-semibold">{preview.durationMs.toFixed(1)} ms</p>
                </div>
              </div>

              {preview.sampleRows.length > 0 ? (
                <div className="overflow-auto rounded-lg border">
                  <table className="min-w-full divide-y divide-border text-sm">
                    <thead className="bg-muted/30 text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <tr>
                        {header.map((column, idx) => (
                          <th key={idx} className="px-3 py-2 whitespace-nowrap font-medium">
                            {column || `Column ${idx + 1}`}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border bg-background/60">
                      {preview.sampleRows.map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {row.map((value, colIdx) => (
                            <td key={colIdx} className="px-3 py-2 whitespace-nowrap">
                              {value || '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground border rounded-lg p-4 text-center">
                  No rows parsed from the selected file.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function formatRange(min: number | null, max: number | null, unit?: string) {
  if (min === null || max === null) {
    return '-'
  }
  const suffix = unit ? ` ${unit}` : ''
  return `${min.toLocaleString()} - ${max.toLocaleString()}${suffix}`
}
