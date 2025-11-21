import { Loader2, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { InfoTooltip } from "@/components/ui/info-tooltip"
import { Separator } from "@/components/ui/separator"
import type { ParamsStore } from "@/lib/stores/paramsStore"

interface PreprocessControlsProps {
  params: ParamsStore['params']
  updateParam: ParamsStore['updateParam']
  onApplyFilter: () => void
  onDetectPeaks: () => void
  onAutoThreshold: () => void
  onAutoCutoff: () => void
  loading: boolean
  detecting: boolean
  progress: number
  resultId: string | null
  prominenceInput: string
  setProminenceInput: (val: string) => void
}

export function PreprocessControls({
  params,
  updateParam,
  onApplyFilter,
  onDetectPeaks,
  onAutoThreshold,
  onAutoCutoff,
  loading,
  detecting,
  progress,
  resultId,
  prominenceInput,
  setProminenceInput,
}: PreprocessControlsProps) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Process & Detect</h2>
        <p className="text-sm text-muted-foreground">
          Filter signal and detect peaks
        </p>
      </div>

      <Accordion type="multiple" defaultValue={["filter", "detect"]} className="w-full">
        {/* Filter Section */}
        <AccordionItem value="filter" className="border-none">
          <AccordionTrigger className="py-2 hover:no-underline">
            <div className="flex items-center gap-2 text-sm font-medium">
              Filter Signal
            </div>
          </AccordionTrigger>
          <AccordionContent className="pt-2 pb-0 space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-medium flex items-center gap-1">
                Filter Type
                <InfoTooltip content="Digital filters remove noise from the signal." />
              </label>
              <select 
                className="w-full px-3 py-2 rounded-md border bg-transparent text-xs shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                value={params.filter_type}
                onChange={(e) => updateParam('filter_type', e.target.value as any)}
              >
                <option value="none">None</option>
                <option value="butterworth">Butterworth</option>
                <option value="savgol">Savitzky-Golay</option>
              </select>
            </div>

            {params.filter_type === 'butterworth' && (
              <div className="space-y-3 animate-in slide-in-from-top-2 fade-in duration-200">
                <div className="space-y-2">
                  <label className="text-xs font-medium flex items-center gap-1">
                    Cutoff Frequency (Hz)
                    <InfoTooltip content="Frequencies above this value are attenuated." />
                  </label>
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      step="1"
                      value={params.filter_cutoff_freq}
                      onChange={(e) => updateParam('filter_cutoff_freq', Math.round(parseFloat(e.target.value) || 0))}
                      className="text-xs"
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={onAutoCutoff}
                      disabled={!resultId}
                      className="h-9 px-3"
                    >
                      Auto
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium flex items-center gap-1">
                    Order
                    <InfoTooltip content="Steepness of the filter roll-off." />
                  </label>
                  <Input
                    type="number"
                    value={params.butter_order}
                    onChange={(e) => updateParam('butter_order', parseInt(e.target.value))}
                    className="text-xs"
                  />
                </div>
              </div>
            )}

            {params.filter_type === 'savgol' && (
              <div className="space-y-3 animate-in slide-in-from-top-2 fade-in duration-200">
                <div className="space-y-2">
                  <label className="text-xs font-medium flex items-center gap-1">
                    Window Length
                    <InfoTooltip content="Number of points used for the polynomial fit. Must be odd." />
                  </label>
                  <Input
                    type="number"
                    value={params.savgol_window}
                    onChange={(e) => updateParam('savgol_window', parseInt(e.target.value))}
                    className="text-xs"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium flex items-center gap-1">
                    Polynomial Order
                    <InfoTooltip content="Degree of the polynomial." />
                  </label>
                  <Input
                    type="number"
                    value={params.savgol_polyorder}
                    onChange={(e) => updateParam('savgol_polyorder', parseInt(e.target.value))}
                    className="text-xs"
                  />
                </div>
              </div>
            )}

            <Button 
              className="w-full" 
              onClick={onApplyFilter}
              disabled={loading || !resultId || params.filter_type === 'none'}
              size="sm"
            >
              {loading ? (
                <>
                  <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                  {progress > 0 ? `${progress}%` : 'Processing...'}
                </>
              ) : (
                'Apply Filter'
              )}
            </Button>
          </AccordionContent>
        </AccordionItem>
        
        <Separator className="my-2" />

        {/* Detect Section */}
        <AccordionItem value="detect" className="border-none">
          <AccordionTrigger className="py-2 hover:no-underline">
            <div className="flex items-center gap-2 text-sm font-medium">
              Detect Peaks
            </div>
          </AccordionTrigger>
          <AccordionContent className="pt-2 pb-0 space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2 col-span-2">
                <label className="text-xs font-medium flex items-center gap-1">
                  Prominence
                  <InfoTooltip content="Vertical distance between the peak and its lowest contour line." />
                </label>
                <Input
                  type="number"
                  value={prominenceInput}
                  onChange={(e) => {
                    const inputValue = e.target.value
                    setProminenceInput(inputValue)
                    if (inputValue !== '' && inputValue !== '-') {
                      const value = parseFloat(inputValue)
                      if (!isNaN(value) && value >= 0 && isFinite(value)) {
                        updateParam('prominence_threshold', value)
                      }
                    }
                  }}
                  onBlur={(e) => {
                    const value = parseFloat(e.target.value)
                    if (isNaN(value) || value < 0 || !isFinite(value)) {
                      const defaultValue = 20
                      setProminenceInput(defaultValue.toString())
                      updateParam('prominence_threshold', defaultValue)
                    } else {
                      setProminenceInput(value.toString())
                    }
                  }}
                  step="0.1"
                  min="0"
                  className="text-xs"
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-xs font-medium flex items-center gap-1">
                  Min Distance
                  <InfoTooltip content="Minimum horizontal distance (samples) between peaks." />
                </label>
                <Input
                  type="number"
                  value={params.distance}
                  onChange={(e) => updateParam('distance', parseInt(e.target.value))}
                  className="text-xs"
                />
              </div>
              
               <div className="space-y-2">
                <label className="text-xs font-medium flex items-center gap-1">
                  Rel Height (%)
                  <InfoTooltip content="Relative height for width measurement." />
                </label>
                <div className="relative">
                  <Input
                    type="number"
                    value={Number((params.rel_height * 100).toFixed(1))}
                    onChange={(e) => updateParam('rel_height', parseFloat(e.target.value) / 100)}
                    step="0.5"
                    min="0"
                    max="100"
                    className="text-xs pr-6"
                  />
                   <span className="absolute inset-y-0 right-2 flex items-center text-[10px] text-muted-foreground">%</span>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-medium flex items-center gap-1">
                  Width Min (ms)
                </label>
                <Input
                  type="number"
                  value={params.width_ms.split(',')[0]}
                  onChange={(e) => {
                    const [, max] = params.width_ms.split(',')
                    updateParam('width_ms', `${e.target.value},${max || 200}`)
                  }}
                  step="0.1"
                  min="0.01"
                  className="text-xs"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium flex items-center gap-1">
                  Width Max (ms)
                </label>
                <Input
                  type="number"
                  value={params.width_ms.split(',')[1]}
                  onChange={(e) => {
                    const [min] = params.width_ms.split(',')
                    updateParam('width_ms', `${min || 0.1},${e.target.value}`)
                  }}
                  step="1"
                  min="0.1"
                  className="text-xs"
                />
              </div>
              
               <div className="space-y-2 col-span-2">
                <label className="text-xs font-medium flex items-center gap-1">
                  Prominence Ratio (%)
                  <InfoTooltip content="Ratio of prominence to amplitude." />
                </label>
                 <div className="relative">
                  <Input
                    type="number"
                    value={Number((params.prominence_ratio * 100).toFixed(1))}
                    onChange={(e) => updateParam('prominence_ratio', parseFloat(e.target.value) / 100)}
                    step="0.5"
                    min="0"
                    max="100"
                    className="text-xs pr-6"
                  />
                  <span className="absolute inset-y-0 right-2 flex items-center text-[10px] text-muted-foreground">%</span>
                </div>
              </div>
            </div>

            <div className="flex gap-2 pt-2">
              <Button 
                className="flex-1"
                onClick={onDetectPeaks}
                disabled={detecting || !resultId}
                size="sm"
              >
                {detecting ? (
                  <>
                    <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                    Detecting...
                  </>
                ) : (
                  'Detect Peaks'
                )}
              </Button>
              <Button
                variant="outline"
                onClick={onAutoThreshold}
                disabled={!resultId}
                size="sm"
                className="px-3"
              >
                Auto
              </Button>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  )
}

