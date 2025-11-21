import { useRef } from "react"
import { Loader2, FileUp, Upload, Settings2, FileText, RotateCcw, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { BlockMath } from "react-katex"
import "katex/dist/katex.min.css"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { InfoTooltip } from "@/components/ui/info-tooltip"

interface LoadControlsProps {
  selectedFiles: File[]
  loading: boolean
  timeResolutionMs: number
  onTimeResolutionChange: (val: number) => void
  photonCorrection: {
    applyCorrection: boolean
    deadTimeNs: number
  }
  setApplyCorrection: (val: boolean) => void
  setDeadTimeNs: (val: number) => void
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void
  onSelectClick: () => void
  onLoadClick: () => void
}

export function LoadControls({
  selectedFiles,
  loading,
  timeResolutionMs,
  onTimeResolutionChange,
  photonCorrection,
  setApplyCorrection,
  setDeadTimeNs,
  onFileSelect,
  onSelectClick,
  onLoadClick,
}: LoadControlsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Load Data</h2>
        <p className="text-sm text-muted-foreground">
          Select files to begin analysis
        </p>
      </div>

      <div className="space-y-4">
        {/* File Selection Area */}
        <div className="space-y-3">
           <Button 
            className="w-full" 
            variant="outline"
            onClick={onSelectClick}
            disabled={loading}
          >
            <Upload className="h-4 w-4 mr-2" />
            Select Files
          </Button>

          {selectedFiles.length > 0 && (
            <div className="rounded-lg border bg-muted/30 p-3 space-y-2">
              <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
                <span>{selectedFiles.length} file(s) selected</span>
              </div>
              <div className="max-h-[120px] overflow-y-auto space-y-1 pr-1 custom-scrollbar">
                {selectedFiles.map((file, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-xs text-foreground/90 bg-background/50 px-2 py-1.5 rounded-md border border-transparent hover:border-border transition-colors">
                    <FileText className="h-3 w-3 shrink-0 text-muted-foreground" />
                    <span className="truncate">{file.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <Button 
            className="w-full" 
            onClick={onLoadClick}
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
        </div>

        <Separator />

        {/* Settings Accordion */}
        <Accordion type="multiple" defaultValue={["settings"]} className="w-full">
          <AccordionItem value="settings" className="border-none">
            <AccordionTrigger className="py-2 hover:no-underline">
               <div className="flex items-center gap-2 text-sm font-medium">
                 <Settings2 className="h-4 w-4 text-muted-foreground" />
                 File Settings
               </div>
            </AccordionTrigger>
            <AccordionContent className="pt-2 pb-0 space-y-6">
              <div className="space-y-2">
                <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                  Time Resolution (ms)
                  <InfoTooltip content="Sampling interval of the data. Incorrect values will distort time and throughput calculations." />
                </label>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    placeholder="0.1"
                    value={timeResolutionMs}
                    onChange={(e) => onTimeResolutionChange(parseFloat(e.target.value))}
                    step="0.01"
                    min="0.001"
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => onTimeResolutionChange(0.1)}
                    title="Reset to default (0.1 ms)"
                    className="shrink-0"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-[10px] text-muted-foreground">
                  Default is 0.1 ms. Verify this matches your instrument settings.
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <label className="text-xs font-medium text-muted-foreground">Dead-time Correction</label>
                    <p className="text-[10px] text-muted-foreground">Compensate for detector blindness</p>
                  </div>
                  <Switch
                    checked={photonCorrection.applyCorrection}
                    onCheckedChange={setApplyCorrection}
                  />
                </div>
                
                {photonCorrection.applyCorrection && (
                  <div className="space-y-3 animate-in slide-in-from-top-2 fade-in duration-200 bg-muted/30 p-3 rounded-lg">
                    <div className="space-y-2">
                      <label className="text-xs font-medium text-muted-foreground">Dead time (ns)</label>
                      <Input
                        type="number"
                        value={photonCorrection.deadTimeNs}
                        onChange={(e) => setDeadTimeNs(parseFloat(e.target.value) || 43.0)}
                        step="1"
                        min="1"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <p className="text-[10px] text-muted-foreground leading-relaxed">
                        Photon counters have a "dead time" <span className="font-mono">T_D</span> after each detection where they cannot detect new photons.
                        This correction scales the measured rate <span className="font-mono">R</span> to the true rate using the formula:
                      </p>
                      <div className="text-xs text-muted-foreground bg-background p-2 rounded-md text-center border">
                         <BlockMath math="\text{Rate}_{true} = \frac{\text{Rate}_{measured}}{1 - \text{Rate}_{measured} \times T_D}" />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </div>
  )
}
