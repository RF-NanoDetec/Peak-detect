import { AlertCircle, CheckCircle, ChevronRight, Clock, FileText, FolderOpen, PlayCircle, UploadCloud } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { LoadingState } from "@/components/ui/loading-state"
import { ProtocolInfo, RecentSession } from "@/lib/types"
import type { LocalPreviewResult } from "@/lib/localDataParser"
import { workflowSteps } from "@/components/layout/workflow"

interface LoadViewProps {
  selectedFiles: File[]
  loading: boolean
  previewLoading: boolean
  previewError: string | null
  localPreview: LocalPreviewResult | null
  protocol: ProtocolInfo
  updateProtocol: <K extends keyof ProtocolInfo>(key: K, value: ProtocolInfo[K]) => void
  onSelectFiles: () => void
  onDropFiles: (files: File[]) => void
  onLoadFiles: () => void
  recentFiles: RecentSession[]
  onRecentClick: (session: RecentSession) => void
}

export function LoadView({
  selectedFiles,
  loading,
  previewLoading,
  previewError,
  localPreview,
  protocol,
  updateProtocol,
  onSelectFiles,
  onDropFiles,
  onLoadFiles,
  recentFiles,
  onRecentClick,
}: LoadViewProps) {
  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        {selectedFiles.length === 0 ? (
          <LoadLandingHero
            onSelectFiles={onSelectFiles}
            onDropFiles={onDropFiles}
            recentFiles={recentFiles}
            onRecentClick={onRecentClick}
            loading={loading}
          />
        ) : (
          <LocalPreviewPanel
            loading={previewLoading}
            error={previewError}
            preview={localPreview}
            fileCount={selectedFiles.length}
            onSelectFiles={onSelectFiles}
            onLoadFiles={onLoadFiles}
            loadInProgress={loading}
          />
        )}

        <ProtocolInformationAccordion protocol={protocol} updateProtocol={updateProtocol} />
      </div>
    </div>
  )
}

function LoadLandingHero({
  onSelectFiles,
  onDropFiles,
  recentFiles,
  onRecentClick,
  loading,
}: {
  onSelectFiles: () => void
  onDropFiles: (files: File[]) => void
  recentFiles: RecentSession[]
  onRecentClick: (session: RecentSession) => void
  loading: boolean
}) {
  const validRecents = recentFiles
    .filter((session): session is RecentSession => Array.isArray(session.files) && session.files.length > 0)
    .slice(0, 4)

  return (
    <div className="space-y-8 max-w-3xl mx-auto pt-10">
      <section className="rounded-xl border bg-card shadow-sm overflow-hidden">
        <div className="px-6 pt-6 text-center">
          <h2 className="text-2xl font-semibold tracking-tight">Load Measurement Data</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Import your time-series data files to begin the analysis workflow.
          </p>
        </div>

        <div className="p-6">
          <div
            className="rounded-lg border border-dashed bg-muted/25 px-6 py-10 text-center transition-colors hover:bg-muted/40"
            onDragOver={(event) => {
              event.preventDefault()
              event.dataTransfer.dropEffect = "copy"
            }}
            onDrop={(event) => {
              event.preventDefault()
              const files = Array.from(event.dataTransfer.files).filter((file) =>
                /\.(txt|xls|xlsx)$/i.test(file.name)
              )
              if (files.length > 0) {
                onDropFiles(files)
              }
            }}
          >
            <UploadCloud className="mx-auto h-10 w-10 text-accent" />
            <p className="mt-4 text-sm font-medium">Drag and drop files here</p>
            <p className="mt-1 text-xs text-muted-foreground">or choose files from your computer</p>
            <Button size="lg" onClick={onSelectFiles} disabled={loading} className="mt-5 min-w-[180px]">
              Select Files
            </Button>
            <p className="mt-4 text-[11px] font-mono uppercase tracking-widest text-muted-foreground">
              Accepted formats: .txt, .xls, .xlsx
            </p>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-3 gap-3 text-left max-w-xl mx-auto">
        {workflowSteps.slice(1).map((step) => (
          <div key={step.id} className="p-3 rounded-lg border bg-card/50">
            <div className="font-medium text-sm mb-1 flex items-center gap-2">
              <step.icon className="h-3.5 w-3.5 text-accent" />
              {step.label}
            </div>
            <p className="text-[10px] text-muted-foreground">{step.description}</p>
          </div>
        ))}
      </div>

      {validRecents.length > 0 && (
        <div className="space-y-4 pt-8 border-t">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground uppercase tracking-wider">
            <Clock className="h-4 w-4" />
            RECENT SESSIONS
          </div>

          <div className="grid gap-2">
            {validRecents.map((session, idx) => (
              <button
                key={`${session.files.join('|')}-${session.loadedAt ?? 'legacy'}-${idx}`}
                className="group w-full flex items-center justify-between gap-4 rounded-lg border bg-card/50 px-4 py-3 text-sm hover:bg-accent hover:text-accent-foreground transition-colors text-left"
                onClick={() => onRecentClick(session)}
                disabled={loading}
              >
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2 min-w-0">
                    <FolderOpen className="h-4 w-4 shrink-0 text-muted-foreground group-hover:text-accent-foreground/80" />
                    <span className="truncate font-medium">{formatRecentSessionTitle(session)}</span>
                  </span>
                  <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground group-hover:text-accent-foreground/80">
                    <span>{formatRecentSessionDate(session.loadedAt)}</span>
                    <span aria-hidden="true">/</span>
                    <span>{formatRecentSessionCount(session.files.length)}</span>
                  </span>
                </span>
                <span className="shrink-0 inline-flex items-center gap-1 text-xs font-medium text-muted-foreground group-hover:text-accent-foreground">
                  Open
                  <ChevronRight className="h-4 w-4" />
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function formatRecentSessionTitle(session: RecentSession) {
  if (session.originLabel) return session.originLabel
  if (session.files.length <= 1) return session.files[0] ?? "Untitled session"
  return "Mixed file selection"
}

function formatRecentSessionDate(loadedAt?: string) {
  if (!loadedAt) return "Date unknown"
  const date = new Date(loadedAt)
  if (Number.isNaN(date.getTime())) return "Date unknown"
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date)
}

function formatRecentSessionCount(fileCount: number) {
  return `${fileCount.toLocaleString()} ${fileCount === 1 ? "file" : "files"}`
}

function LocalPreviewPanel({
  loading,
  error,
  preview,
  fileCount,
  onSelectFiles,
  onLoadFiles,
  loadInProgress,
}: {
  loading: boolean
  error: string | null
  preview: LocalPreviewResult | null
  fileCount: number
  onSelectFiles: () => void
  onLoadFiles: () => void
  loadInProgress: boolean
}) {
  const header =
    preview && preview.columns.length > 0
      ? preview.columns
      : preview?.sampleRows[0]?.map((_, idx) => `Column ${idx + 1}`) ?? []

  if (loading) {
    return (
      <div className="rounded-xl border bg-card p-6 min-h-[320px] flex items-center justify-center">
        <LoadingState text="Parsing preview with uDSV..." />
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-xl border bg-card p-6 min-h-[320px] flex items-center justify-center">
        <div className="max-w-md text-center">
          <AlertCircle className="mx-auto h-10 w-10 text-muted-foreground" />
          <h3 className="mt-4 text-lg font-semibold tracking-tight">Preview unavailable</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            {error}
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            The files are still selected. You can load them now or choose a different set.
          </p>
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            <Button variant="outline" size="sm" onClick={onSelectFiles} disabled={loadInProgress}>
              <FolderOpen className="h-4 w-4 mr-2" />
              Choose Different Files
            </Button>
            <Button onClick={onLoadFiles} size="sm" disabled={loadInProgress}>
              {loadInProgress ? (
                "Loading..."
              ) : (
                <>
                  <PlayCircle className="h-4 w-4 mr-2" />
                  Load and Continue
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    )
  }

  if (!preview) return null

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold">Local Data Preview</h3>
          <p className="text-sm text-muted-foreground">
            Showing first file ({fileCount} selected)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={onSelectFiles} size="sm" disabled={loadInProgress}>
            <FolderOpen className="h-4 w-4 mr-2" />
            Choose Different Files
          </Button>
          <Button onClick={onLoadFiles} size="sm" disabled={loadInProgress}>
            {loadInProgress ? (
              "Loading..."
            ) : (
              <>
                <PlayCircle className="h-4 w-4 mr-2" />
                Load and Continue
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="rounded-xl border bg-card shadow-sm overflow-hidden">
        <div className="p-4 border-b bg-muted/30 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground" />
              {preview.fileName}
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {preview.rowsParsed.toLocaleString()} rows • {preview.durationMs.toFixed(1)} ms
            </p>
          </div>
        </div>

        <div className="border-b bg-background/60 px-4 py-2">
          <div className="flex items-center gap-2 text-xs text-accent">
            <CheckCircle className="h-3.5 w-3.5" />
            Preview ready. Review the ranges, then load the files to continue to detection.
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 border-b divide-x">
          <div className="p-3 text-center">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Points</p>
            <p className="text-sm font-medium">{preview.rowsParsed.toLocaleString()}</p>
          </div>
          <div className="p-3 text-center">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Time Range</p>
            <p className="text-sm font-medium">
              {formatRange(preview.stats.minTime, preview.stats.maxTime, "s")}
            </p>
          </div>
          <div className="p-3 text-center">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Amplitude</p>
            <p className="text-sm font-medium">
              {formatRange(preview.stats.minAmplitude, preview.stats.maxAmplitude)}
            </p>
          </div>
          <div className="p-3 text-center">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Duration</p>
            <p className="text-sm font-medium">{preview.durationMs.toFixed(1)} ms</p>
          </div>
        </div>

        <div className="overflow-auto max-h-[400px]">
          <table className="min-w-full divide-y divide-border text-sm">
            <thead className="bg-muted/50 text-left text-xs uppercase tracking-wide text-muted-foreground sticky top-0">
              <tr>
                {header.map((column, idx) => (
                  <th key={idx} className="px-4 py-2 whitespace-nowrap font-medium">
                    {column || `Column ${idx + 1}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border bg-card">
              {preview.sampleRows.map((row, rowIdx) => (
                <tr key={rowIdx} className="hover:bg-muted/50 transition-colors">
                  {row.map((value, colIdx) => (
                    <td key={colIdx} className="px-4 py-2 whitespace-nowrap text-muted-foreground">
                      {value || "-"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function ProtocolInformationAccordion({
  protocol,
  updateProtocol,
}: {
  protocol: ProtocolInfo
  updateProtocol: <K extends keyof ProtocolInfo>(key: K, value: ProtocolInfo[K]) => void
}) {
  return (
    <Card>
      <Accordion type="single" collapsible className="w-full" defaultValue="protocol">
        <AccordionItem value="protocol" className="border-0">
          <AccordionTrigger className="px-6 hover:no-underline">
            <div className="text-left">
              <CardTitle className="text-base">Protocol Information</CardTitle>
              <CardDescription className="mt-1">
                Experiment metadata attached to the dataset
              </CardDescription>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-6 pb-6">
            <div className="space-y-4 pt-2">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Measurement Date</label>
                  <Input
                    type="date"
                    value={protocol.measurement_date || ""}
                    onChange={(e) => updateProtocol("measurement_date", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Start Time</label>
                  <Input
                    type="time"
                    value={protocol.start_time || ""}
                    onChange={(e) => updateProtocol("start_time", e.target.value)}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Setup</label>
                <Input
                  placeholder="e.g., Prototype, Old Ladom"
                  value={protocol.setup || ""}
                  onChange={(e) => updateProtocol("setup", e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Sample Number</label>
                  <Input
                    placeholder="Sample ID"
                    value={protocol.sample_number || ""}
                    onChange={(e) => updateProtocol("sample_number", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Particle</label>
                  <Input
                    placeholder="Particle type"
                    value={protocol.particle || ""}
                    onChange={(e) => updateProtocol("particle", e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Concentration</label>
                  <Input
                    placeholder="Particle concentration"
                    value={protocol.concentration || ""}
                    onChange={(e) => updateProtocol("concentration", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Buffer</label>
                  <Input
                    placeholder="Buffer solution"
                    value={protocol.buffer || ""}
                    onChange={(e) => updateProtocol("buffer", e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Buffer Conc.</label>
                  <Input
                    placeholder="Buffer concentration"
                    value={protocol.buffer_concentration || ""}
                    onChange={(e) => updateProtocol("buffer_concentration", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">ND Filter</label>
                  <Input
                    placeholder="Filter value"
                    value={protocol.nd_filter || ""}
                    onChange={(e) => updateProtocol("nd_filter", e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Laser Power</label>
                  <Input
                    placeholder="Power setting"
                    value={protocol.laser_power || ""}
                    onChange={(e) => updateProtocol("laser_power", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Stamp</label>
                  <Input
                    placeholder="e.g., triple-block"
                    value={protocol.stamp || ""}
                    onChange={(e) => updateProtocol("stamp", e.target.value)}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                  <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Notes</label>
                  <Input
                    placeholder="Additional observations"
                    value={protocol.notes || ""}
                    onChange={(e) => updateProtocol("notes", e.target.value)}
                  />
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Card>
  )
}

function formatRange(min: number | null, max: number | null, unit?: string) {
  if (min === null || max === null) return "-"
  const suffix = unit ? ` ${unit}` : ""
  return `${min.toLocaleString()} - ${max.toLocaleString()}${suffix}`
}
