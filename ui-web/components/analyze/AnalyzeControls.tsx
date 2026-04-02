import { BarChart3, Download } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { InfoTooltip } from "@/components/ui/info-tooltip"
import { Button } from "@/components/ui/button"
import Link from "next/link"

interface AnalyzeControlsProps {
  hasResults: boolean
  binWidthSeconds: number
  setBinWidthSeconds: (val: number) => void
  rollingMeanWindow: number
  setRollingMeanWindow: (val: number) => void
}

export function AnalyzeControls({
  hasResults,
  binWidthSeconds,
  setBinWidthSeconds,
  rollingMeanWindow,
  setRollingMeanWindow,
}: AnalyzeControlsProps) {
  if (!hasResults) {
    return (
      <div className="space-y-4">
         <div>
            <h2 className="text-lg font-semibold tracking-tight">Analyze & Export</h2>
            <p className="text-sm text-muted-foreground">
              Time series analysis and data export
            </p>
          </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Analyze Results</h2>
        <p className="text-sm text-muted-foreground">
          Time series analysis of detected peaks
        </p>
      </div>

      <Accordion type="multiple" defaultValue={["analysis"]} className="w-full">
        <AccordionItem value="analysis" className="border-none">
           <AccordionTrigger className="py-2 hover:no-underline">
              <div className="flex items-center gap-2 text-sm font-medium">
                Visualization Settings
              </div>
           </AccordionTrigger>
           <AccordionContent className="pt-2 pb-0 space-y-6">
             <div className="space-y-3">
               <div className="flex items-center justify-between">
                  <label className="text-xs font-medium flex items-center gap-1">
                    Throughput Binning
                    <InfoTooltip content="Time window for calculating peak throughput (peaks per second)." />
                  </label>
                  <span className="text-xs font-medium text-muted-foreground bg-muted/50 px-2 py-0.5 rounded">
                    {binWidthSeconds.toFixed(1)} s
                  </span>
               </div>
               <input
                  type="range"
                  min={0.5}
                  max={20}
                  step={0.5}
                  value={binWidthSeconds}
                  onChange={(e) => setBinWidthSeconds(parseFloat(e.target.value))}
                  className="w-full"
                />
             </div>
             
             <div className="space-y-3">
               <div className="flex items-center justify-between">
                  <label className="text-xs font-medium flex items-center gap-1">
                    Rolling Mean Window
                    <InfoTooltip content="Number of points for rolling average smoothing of metrics." />
                  </label>
                  <span className="text-xs font-medium text-muted-foreground bg-muted/50 px-2 py-0.5 rounded">
                    {rollingMeanWindow} pts
                  </span>
               </div>
               <input
                  type="range"
                  min={2}
                  max={200}
                  step={1}
                  value={rollingMeanWindow}
                  onChange={(e) => setRollingMeanWindow(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-[10px] text-muted-foreground px-1">
                  <span>Low smoothing</span>
                  <span>High smoothing</span>
                </div>
             </div>
           </AccordionContent>
        </AccordionItem>
      </Accordion>

      <Button variant="outline" className="w-full" asChild>
        <Link href="/export">
          <Download className="h-4 w-4 mr-2" />
          Export Data
        </Link>
      </Button>
    </div>
  )
}

