import { AlertCircle, CheckCircle, ChevronRight, Clock, Database, FileText, FolderOpen, UploadCloud } from "lucide-react"
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
  loadProgress: LoadProgress | null
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

export interface LoadProgress {
  phase: "uploading" | "processing" | "preview" | "complete"
  loadedFiles: number
  totalFiles: number
  percent: number
}

export function LoadView({
  selectedFiles,
  loading,
  loadProgress,
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
            selectedFiles={selectedFiles}
            fileCount={selectedFiles.length}
            onSelectFiles={onSelectFiles}
            onLoadFiles={onLoadFiles}
            loadInProgress={loading}
            loadProgress={loadProgress}
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
                    <FileText className="h-4 w-4 shrink-0 text-muted-foreground group-hover:text-accent-foreground/80" />
                    <span className="truncate font-medium">{formatRecentSessionTitle(session)}</span>
                  </span>
                  <span className="mt-1 flex items-center gap-2 min-w-0 text-xs text-muted-foreground group-hover:text-accent-foreground/80">
                    <FolderOpen className="h-3.5 w-3.5 shrink-0" />
                    <span className="truncate">{formatRecentSessionFolder(session)}</span>
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
  if (session.files.length <= 1) return session.files[0] ?? "Untitled session"
  return `${session.files[0]} + ${session.files.length - 1}`
}

function formatRecentSessionFolder(session: RecentSession) {
  if (session.originKind === "folder" && session.originLabel) {
    return session.originLabel
  }
  return "Folder unavailable from browser picker"
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
  selectedFiles,
  fileCount,
  onSelectFiles,
  onLoadFiles,
  loadInProgress,
  loadProgress,
}: {
  loading: boolean
  error: string | null
  preview: LocalPreviewResult | null
  selectedFiles: File[]
  fileCount: number
  onSelectFiles: () => void
  onLoadFiles: () => void
  loadInProgress: boolean
  loadProgress: LoadProgress | null
}) {
  if (loading) {
    return (
      <div className="rounded-xl border bg-card p-6 min-h-[320px] flex items-center justify-center">
        <LoadingState text="Checking selected files..." />
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
            <Button variant="outline" size="sm" onClick={onSelectFiles} disabled={loadInProgress} className="min-w-0">
              <FolderOpen className="h-4 w-4 mr-2" />
              Choose Different Files
            </Button>
            <Button onClick={onLoadFiles} size="sm" disabled={loadInProgress} className="min-w-0">
              <LoadActionIcon progress={loadProgress} loading={loadInProgress} />
              <span className="ml-2 truncate">{loadInProgress ? getLoadProgressLabel(loadProgress) : "Load files"}</span>
            </Button>
          </div>
        </div>
      </div>
    )
  }

  if (!preview) return null
  const summary = getSelectedFilesSummary(selectedFiles, preview)
  const visibleFiles = selectedFiles.slice(0, 8)
  const remainingFiles = Math.max(0, selectedFiles.length - visibleFiles.length)

  return (
    <div className="rounded-xl border bg-card shadow-sm overflow-hidden">
      <div className="p-5 border-b bg-muted/20">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-medium text-accent">
              <CheckCircle className="h-4 w-4 shrink-0" />
              Files ready
            </div>
            <h3 className="mt-2 text-lg font-semibold tracking-tight">
              {fileCount.toLocaleString()} {fileCount === 1 ? "file selected" : "files selected"}
            </h3>
            <p className="mt-1 truncate text-sm text-muted-foreground" title={summary.origin}>
              {summary.origin}
            </p>
          </div>
          <div className="grid w-full grid-cols-1 gap-2 sm:grid-cols-2 lg:flex lg:w-auto lg:shrink-0">
            <Button variant="outline" onClick={onSelectFiles} size="sm" disabled={loadInProgress} className="min-w-0">
              <FolderOpen className="h-4 w-4 mr-2 shrink-0" />
              <span className="truncate">Choose different files</span>
            </Button>
            <Button onClick={onLoadFiles} size="sm" disabled={loadInProgress} className="min-w-0">
              <LoadActionIcon progress={loadProgress} loading={loadInProgress} />
              <span className="ml-2 truncate">{loadInProgress ? getLoadProgressLabel(loadProgress) : "Load files"}</span>
            </Button>
          </div>
        </div>
      </div>

      {loadInProgress && (
        <div className="border-b px-5 py-3">
          <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
            <span>{getLoadProgressDetail(loadProgress)}</span>
            <span className="font-medium text-foreground">{loadProgress?.percent ?? 0}%</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-accent transition-all duration-300"
              style={{ width: `${loadProgress?.percent ?? 0}%` }}
            />
          </div>
        </div>
      )}

      <div className="grid gap-0 md:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-w-0 border-b md:border-b-0 md:border-r">
          <div className="px-5 py-4">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
              <FileText className="h-3.5 w-3.5" />
              Selected files
            </div>
            <div className="mt-3 divide-y rounded-lg border bg-background/50">
              {visibleFiles.map((file) => (
                <div key={`${file.name}-${file.size}-${file.lastModified}`} className="flex min-w-0 items-center justify-between gap-3 px-3 py-2.5">
                  <span className="truncate text-sm font-medium" title={file.name}>{file.name}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">{formatFileSize(file.size)}</span>
                </div>
              ))}
              {remainingFiles > 0 && (
                <div className="px-3 py-2.5 text-sm text-muted-foreground">
                  + {remainingFiles.toLocaleString()} more {remainingFiles === 1 ? "file" : "files"}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="px-5 py-4">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
            <Database className="h-3.5 w-3.5" />
            Import check
          </div>
          <dl className="mt-3 space-y-3 text-sm">
            <div>
              <dt className="text-xs text-muted-foreground">First file</dt>
              <dd className="mt-0.5 truncate font-medium" title={preview.fileName}>{preview.fileName}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Recorded dates</dt>
              <dd className="mt-0.5 font-medium">{summary.dateRange}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">File size</dt>
              <dd className="mt-0.5 font-medium">{summary.totalSize}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Detected columns</dt>
              <dd className="mt-0.5 truncate font-medium" title={summary.columns}>{summary.columns}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Signal range</dt>
              <dd className="mt-0.5 font-medium">{formatRange(preview.stats.minAmplitude, preview.stats.maxAmplitude)}</dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  )
}

function LoadActionIcon({ progress, loading }: { progress: LoadProgress | null; loading: boolean }) {
  if (!loading) {
    return <UploadCloud className="h-4 w-4 shrink-0" />
  }

  const percent = Math.max(0, Math.min(100, progress?.percent ?? 0))
  const radius = 8
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percent / 100) * circumference

  return (
    <span className="relative h-4 w-4 shrink-0" aria-hidden="true">
      <svg className="h-4 w-4 -rotate-90" viewBox="0 0 20 20">
        <circle className="stroke-current opacity-25" cx="10" cy="10" r={radius} fill="none" strokeWidth="2" />
        <circle
          className="stroke-current transition-[stroke-dashoffset] duration-300"
          cx="10"
          cy="10"
          r={radius}
          fill="none"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <span className="absolute left-1/2 top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-current" />
    </span>
  )
}

function getLoadProgressLabel(progress: LoadProgress | null) {
  if (!progress) return "Loading..."
  if (progress.phase === "processing") return "Processing..."
  if (progress.phase === "preview") return "Opening..."
  if (progress.phase === "complete") return "Loaded"
  return `Loading ${progress.loadedFiles}/${progress.totalFiles}`
}

function getLoadProgressDetail(progress: LoadProgress | null) {
  if (!progress) return "Preparing selected files..."
  if (progress.phase === "processing") return "Files uploaded. Preparing the dataset..."
  if (progress.phase === "preview") return "Preparing the first chart..."
  if (progress.phase === "complete") return "Dataset loaded."
  return `Uploading ${progress.loadedFiles.toLocaleString()} of ${progress.totalFiles.toLocaleString()} files`
}

function getSelectedFilesSummary(files: File[], preview: LocalPreviewResult) {
  return {
    origin: getSelectionOrigin(files),
    dateRange: formatFileDateRange(files),
    totalSize: formatFileSize(files.reduce((total, file) => total + file.size, 0)),
    columns: preview.columns.length > 0 ? preview.columns.join(", ") : "No named columns detected",
  }
}

function getSelectionOrigin(files: File[]) {
  const directories = files.map(getFileDirectory).filter((dir): dir is string => Boolean(dir))
  if (directories.length === files.length && directories.length > 0) {
    const commonDirectory = getCommonPathPrefix(directories)
    if (commonDirectory) return `Folder: ${commonDirectory}`
  }

  const filenamePattern = getSharedFilenamePattern(files.map((file) => file.name))
  if (filenamePattern) return `Filename group: ${filenamePattern}`

  return "Local file selection"
}

function getFileDirectory(file: File) {
  const nativePath = (file as File & { path?: string }).path
  if (nativePath) return getParentDirectory(nativePath)
  const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath
  return relativePath ? getParentDirectory(relativePath) : null
}

function getParentDirectory(path: string) {
  const pathParts = path.replace(/\\/g, "/").split("/").filter(Boolean)
  if (pathParts.length <= 1) return null
  pathParts.pop()
  return pathParts.join("/")
}

function getCommonPathPrefix(paths: string[]) {
  const splitPaths = paths.map((path) => path.replace(/\\/g, "/").split("/").filter(Boolean))
  const shortestLength = Math.min(...splitPaths.map((parts) => parts.length))
  const commonParts: string[] = []

  for (let idx = 0; idx < shortestLength; idx += 1) {
    const candidate = splitPaths[0][idx]
    if (!splitPaths.every((parts) => parts[idx] === candidate)) break
    commonParts.push(candidate)
  }

  return commonParts.length > 0 ? commonParts.join("/") : null
}

function getSharedFilenamePattern(fileNames: string[]) {
  if (fileNames.length < 2) return null
  const stems = fileNames.map((name) => name.replace(/\.[^.]+$/, ""))
  const shortest = stems.reduce((a, b) => (a.length <= b.length ? a : b), stems[0])
  let prefixLength = 0

  while (
    prefixLength < shortest.length &&
    stems.every((stem) => stem[prefixLength]?.toLowerCase() === shortest[prefixLength].toLowerCase())
  ) {
    prefixLength += 1
  }

  const prefix = shortest.slice(0, prefixLength).replace(/[\s._-]*\d*[\s._-]*$/, "").trim()
  return prefix.length >= 4 ? prefix : null
}

function formatFileDateRange(files: File[]) {
  const timestamps = files.map((file) => file.lastModified).filter((value) => Number.isFinite(value) && value > 0)
  if (timestamps.length === 0) return "Date unavailable"
  const min = Math.min(...timestamps)
  const max = Math.max(...timestamps)
  const formatter = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" })
  if (new Date(min).toDateString() === new Date(max).toDateString()) {
    return formatter.format(min)
  }
  return `${formatter.format(min)} - ${formatter.format(max)}`
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB"]
  const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** unitIndex
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`
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
