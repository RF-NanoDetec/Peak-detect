import { FolderOpen, Upload, Clock, ChevronRight, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { EmptyState } from "@/components/ui/empty-state"
import { LoadingState } from "@/components/ui/loading-state"
import { ProtocolInfo } from "@/lib/types"
import type { LocalPreviewResult } from "@/lib/localDataParser"
import { ReactNode } from "react"

interface LoadViewProps {
  selectedFiles: File[]
  loading: boolean
  previewLoading: boolean
  previewError: string | null
  localPreview: LocalPreviewResult | null
  protocol: ProtocolInfo
  updateProtocol: <K extends keyof ProtocolInfo>(key: K, value: ProtocolInfo[K]) => void
  onSelectFiles: () => void
  recentFiles: string[][]
  onRecentClick: (fileGroup: string[]) => void
  renderRecentLabel: (fileGroup: string[]) => ReactNode
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
  recentFiles,
  onRecentClick,
  renderRecentLabel,
}: LoadViewProps) {
  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        {selectedFiles.length === 0 ? (
          <LoadLandingHero
            onSelectFiles={onSelectFiles}
            recentFiles={recentFiles}
            onRecentClick={onRecentClick}
            loading={loading}
            renderRecentLabel={renderRecentLabel}
          />
        ) : (
          <LocalPreviewPanel
            loading={previewLoading}
            error={previewError}
            preview={localPreview}
            fileCount={selectedFiles.length}
            onSelectFiles={onSelectFiles}
          />
        )}

        <ProtocolInformationAccordion protocol={protocol} updateProtocol={updateProtocol} />
      </div>
    </div>
  )
}

function LoadLandingHero({
  onSelectFiles,
  recentFiles,
  onRecentClick,
  loading,
  renderRecentLabel,
}: {
  onSelectFiles: () => void
  recentFiles: string[][]
  onRecentClick: (fileGroup: string[]) => void
  loading: boolean
  renderRecentLabel: (fileGroup: string[]) => ReactNode
}) {
  const validRecents = recentFiles
    .filter((group): group is string[] => Array.isArray(group) && group.length > 0)
    .slice(0, 4)

  return (
    <div className="space-y-8 max-w-2xl mx-auto pt-12">
      <EmptyState
        title="Load Measurement Data"
        description="Import your time-series data files (.txt, .xls, .xlsx) to begin the analysis workflow."
        action={
          <Button size="lg" onClick={onSelectFiles} disabled={loading} className="min-w-[200px]">
            Select Files
          </Button>
        }
      />

      <div className="grid grid-cols-3 gap-4 text-left max-w-lg mx-auto">
        {[
          { label: "Load", desc: "Import raw data files", color: "bg-primary" },
          { label: "Process", desc: "Filter & detect peaks", color: "bg-muted-foreground/30" },
          { label: "Analyze", desc: "View metrics & export", color: "bg-muted-foreground/30" },
        ].map((step, i) => (
          <div key={i} className="p-3 rounded-lg border bg-card/50">
            <div className="font-medium text-sm mb-1 flex items-center gap-2">
              <div className={`h-1.5 w-1.5 rounded-full ${step.color}`} />
              {step.label}
            </div>
            <p className="text-[10px] text-muted-foreground">{step.desc}</p>
          </div>
        ))}
      </div>

      {validRecents.length > 0 && (
        <div className="space-y-4 pt-8 border-t">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <Clock className="h-4 w-4" />
            Recent Sessions
          </div>

          <div className="grid gap-2">
            {validRecents.map((group, idx) => (
              <button
                key={`${group.join('|')}-${idx}`}
                className="group w-full flex items-center justify-between rounded-lg border bg-card/50 px-4 py-3 text-sm hover:bg-accent hover:text-accent-foreground transition-colors text-left"
                onClick={() => onRecentClick(group)}
                disabled={loading}
              >
                <span className="truncate font-medium">{renderRecentLabel(group)}</span>
                <ChevronRight className="h-4 w-4 text-muted-foreground/50 group-hover:text-muted-foreground transition-colors" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function LocalPreviewPanel({
  loading,
  error,
  preview,
  fileCount,
  onSelectFiles,
}: {
  loading: boolean
  error: string | null
  preview: LocalPreviewResult | null
  fileCount: number
  onSelectFiles: () => void
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
        <EmptyState
          title="Preview failed"
          description={error}
          action={
            <Button variant="outline" size="sm" onClick={onSelectFiles}>
              Try Selecting Files Again
            </Button>
          }
        />
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
        <Button variant="outline" onClick={onSelectFiles} size="sm">
          <FolderOpen className="h-4 w-4 mr-2" />
          Choose Different Files
        </Button>
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
                  <label className="text-xs font-medium text-muted-foreground">Measurement Date</label>
                  <Input
                    type="date"
                    value={protocol.measurement_date || ""}
                    onChange={(e) => updateProtocol("measurement_date", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Start Time</label>
                  <Input
                    type="time"
                    value={protocol.start_time || ""}
                    onChange={(e) => updateProtocol("start_time", e.target.value)}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">Setup</label>
                <Input
                  placeholder="e.g., Prototype, Old Ladom"
                  value={protocol.setup || ""}
                  onChange={(e) => updateProtocol("setup", e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Sample Number</label>
                  <Input
                    placeholder="Sample ID"
                    value={protocol.sample_number || ""}
                    onChange={(e) => updateProtocol("sample_number", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Particle</label>
                  <Input
                    placeholder="Particle type"
                    value={protocol.particle || ""}
                    onChange={(e) => updateProtocol("particle", e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Concentration</label>
                  <Input
                    placeholder="Particle concentration"
                    value={protocol.concentration || ""}
                    onChange={(e) => updateProtocol("concentration", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Buffer</label>
                  <Input
                    placeholder="Buffer solution"
                    value={protocol.buffer || ""}
                    onChange={(e) => updateProtocol("buffer", e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Buffer Conc.</label>
                  <Input
                    placeholder="Buffer concentration"
                    value={protocol.buffer_concentration || ""}
                    onChange={(e) => updateProtocol("buffer_concentration", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">ND Filter</label>
                  <Input
                    placeholder="Filter value"
                    value={protocol.nd_filter || ""}
                    onChange={(e) => updateProtocol("nd_filter", e.target.value)}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Laser Power</label>
                  <Input
                    placeholder="Power setting"
                    value={protocol.laser_power || ""}
                    onChange={(e) => updateProtocol("laser_power", e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Stamp</label>
                  <Input
                    placeholder="e.g., triple-block"
                    value={protocol.stamp || ""}
                    onChange={(e) => updateProtocol("stamp", e.target.value)}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Notes</label>
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
