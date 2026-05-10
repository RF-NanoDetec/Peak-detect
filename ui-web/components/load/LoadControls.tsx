import { Settings2, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { UnitInput } from "@/components/ui/unit-input"
import { InlineMath } from "react-katex"
import "katex/dist/katex.min.css"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { InfoTooltip } from "@/components/ui/info-tooltip"

interface LoadControlsProps {
  timeResolutionMs: number
  onTimeResolutionChange: (val: number) => void
  photonCorrection: {
    applyCorrection: boolean
    deadTimeNs: number
  }
  setApplyCorrection: (val: boolean) => void
  setDeadTimeNs: (val: number) => void
}

export function LoadControls({
  timeResolutionMs,
  onTimeResolutionChange,
  photonCorrection,
  setApplyCorrection,
  setDeadTimeNs,
}: LoadControlsProps) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">File Settings</h2>
        <p className="text-sm text-muted-foreground">
          Adjust import assumptions before loading measurement files.
        </p>
      </div>

      <div className="space-y-4">
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
                  Time Resolution
                  <InfoTooltip content="Sampling interval of the data. Incorrect values will distort time and throughput calculations." />
                </label>
                <div className="flex gap-2">
                  <UnitInput
                    type="number"
                    unit="ms"
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
                    <label className="text-xs font-mono uppercase tracking-widest text-muted-foreground">Dead-time Correction</label>
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
                      <label className="text-xs font-mono tracking-widest text-muted-foreground">Dead time</label>
                      <UnitInput
                        type="number"
                        unit="ns"
                        value={photonCorrection.deadTimeNs}
                        onChange={(e) => setDeadTimeNs(parseFloat(e.target.value) || 43.0)}
                        step="1"
                        min="1"
                      />
                    </div>
                    
                    <p className="text-[10px] text-muted-foreground leading-relaxed">
                      Photon counters have a dead time <InlineMath math="T_D" /> after each detection where they cannot detect new photons. 
                      This correction computes the true rate <InlineMath math="R_t" /> from the measured rate <InlineMath math="R_m" /> by 
                      dividing <InlineMath math="R_m" /> by <InlineMath math="(1 - R_m \times T_D)" />.
                    </p>
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
