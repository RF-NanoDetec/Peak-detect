import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { InfoTooltip } from "@/components/ui/info-tooltip"
import { Separator } from "@/components/ui/separator"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { UnitInput, UnitStepperInput } from "@/components/ui/unit-input"
import type { Parameters, ParamsState } from "@/lib/types"

interface PreprocessControlsProps {
  params: Parameters
  updateParam: ParamsState['updateParam']
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
  onValidateParams: (scope?: "detect" | "filter" | "all", sourceParams?: Parameters) => Parameters
}

function ParamLabel({
  name,
  subscript,
  tooltip,
}: {
  name: string
  subscript?: string
  tooltip?: string
}) {
  return (
    <label className="flex items-center gap-1.5 text-xs font-medium">
      <span className="font-mono text-[11px] tracking-normal text-foreground">
        {name}
        {subscript ? (
          <sub className="relative -bottom-0.5 ml-0.5 text-[9px] leading-none text-muted-foreground">
            {subscript}
          </sub>
        ) : null}
      </span>
      {tooltip ? <InfoTooltip content={tooltip} /> : null}
    </label>
  )
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
  onValidateParams,
}: PreprocessControlsProps) {
  const resolutionMs = params.time_resolution * 1000
  const distanceMs = Number((params.distance * resolutionMs).toFixed(3))
  const updateDistanceMs = (value: number) => {
    const distanceSamples = Math.max(1, Math.round(value / resolutionMs))
    updateParam('distance', distanceSamples)
  }

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
              <Select
                value={params.filter_type}
                onValueChange={(value) => updateParam('filter_type', value as any)}
              >
                <SelectTrigger className="text-xs h-10">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="butterworth">Butterworth</SelectItem>
                  <SelectItem value="savgol">Savitzky-Golay</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {params.filter_type === 'butterworth' && (
              <div className="space-y-3 animate-in slide-in-from-top-2 fade-in duration-200">
                <div className="space-y-2">
                  <label className="text-xs font-medium flex items-center gap-1">
                    Cutoff Frequency
                    <InfoTooltip content="Frequencies above this value are attenuated." />
                  </label>
                  <div className="flex gap-2">
                    <UnitInput
                      type="number"
                      unit="Hz"
                      step="1"
                      value={params.filter_cutoff_freq}
                      onChange={(e) => updateParam('filter_cutoff_freq', Math.round(parseFloat(e.target.value) || 0))}
                      onBlur={(e) => onValidateParams("filter", {
                        ...params,
                        filter_cutoff_freq: Math.round(parseFloat(e.target.value) || 0),
                      })}
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
            <div className="space-y-3">
              <div className="space-y-2">
                <ParamLabel
                  name="prominence"
                  subscript="min"
                  tooltip="Minimum vertical distance between the peak and its lowest contour line."
                />
                <UnitStepperInput
                  value={prominenceInput}
                  unit="counts"
                  onValueChange={(value) => {
                    setProminenceInput(value.toString())
                    updateParam('prominence_threshold', value)
                  }}
                  onChange={(e) => {
                    const inputValue = e.target.value
                    setProminenceInput(inputValue)
                    if (inputValue !== '' && inputValue !== '-') {
                      const value = parseFloat(inputValue)
                      if (!isNaN(value) && value >= 1 && isFinite(value)) {
                        updateParam('prominence_threshold', value)
                      }
                    }
                  }}
                  onBlur={(e) => {
                    const value = parseFloat(e.target.value)
                    if (isNaN(value) || value <= 0 || !isFinite(value)) {
                      const defaultValue = onValidateParams("detect", {
                        ...params,
                        prominence_threshold: value,
                      }).prominence_threshold
                      setProminenceInput(defaultValue.toString())
                      updateParam('prominence_threshold', defaultValue)
                    } else {
                      const corrected = onValidateParams("detect", {
                        ...params,
                        prominence_threshold: value,
                      })
                      if (corrected.prominence_threshold !== value) {
                        setProminenceInput(corrected.prominence_threshold.toString())
                        return
                      }
                      setProminenceInput(value.toString())
                    }
                  }}
                  step={1}
                  min={Number.EPSILON}
                  className="text-xs"
                />
              </div>

              <div className="space-y-2">
                <ParamLabel
                  name="height"
                  subscript="relative"
                  tooltip="Relative height for width measurement."
                />
                <UnitInput
                  type="number"
                  unit="%"
                  value={Number((params.rel_height * 100).toFixed(1))}
                  onChange={(e) => updateParam('rel_height', parseFloat(e.target.value) / 100)}
                  onBlur={(e) => onValidateParams("detect", {
                    ...params,
                    rel_height: parseFloat(e.target.value) / 100,
                  })}
                  step="0.5"
                  min="0"
                  max="100"
                  className="text-xs"
                />
              </div>
            </div>

            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="advanced" className="border-none">
                <AccordionTrigger className="py-1.5 hover:no-underline">
                  <span className="text-xs font-medium text-muted-foreground">Advanced</span>
                </AccordionTrigger>
                <AccordionContent className="pt-2 pb-0 space-y-3">
                  <div className="space-y-2">
                    <ParamLabel name="width" subscript="min" />
                    <UnitStepperInput
                      value={parseFloat(params.width_ms.split(',')[0])}
                      unit="ms"
                      onValueChange={(value) => {
                        const [, max] = params.width_ms.split(',')
                        updateParam('width_ms', `${value},${max || 200}`)
                      }}
                      onBlur={(e) => {
                        const [, max] = params.width_ms.split(',')
                        onValidateParams("detect", {
                          ...params,
                          width_ms: `${e.target.value},${max || 200}`,
                        })
                      }}
                      step={0.1}
                      min={Math.max(0.1, resolutionMs)}
                      className="text-xs"
                    />
                  </div>
                  <div className="space-y-2">
                    <ParamLabel name="width" subscript="max" />
                    <UnitStepperInput
                      value={parseFloat(params.width_ms.split(',')[1])}
                      unit="ms"
                      onValueChange={(value) => {
                        const [min] = params.width_ms.split(',')
                        updateParam('width_ms', `${min || 0.1},${value}`)
                      }}
                      onBlur={(e) => {
                        const [min] = params.width_ms.split(',')
                        onValidateParams("detect", {
                          ...params,
                          width_ms: `${min || 0.1},${e.target.value}`,
                        })
                      }}
                      step={1}
                      min={Math.max(0.1, resolutionMs)}
                      className="text-xs"
                    />
                  </div>
                  <div className="space-y-2">
                    <ParamLabel
                      name="distance"
                      subscript="min"
                      tooltip="Minimum horizontal distance between peaks."
                    />
                    <UnitStepperInput
                      value={distanceMs}
                      unit="ms"
                      onValueChange={updateDistanceMs}
                      onBlur={(e) => {
                        const value = parseFloat(e.target.value)
                        const distanceSamples = Number.isFinite(value)
                          ? Math.round(value / resolutionMs)
                          : params.distance
                        onValidateParams("detect", {
                          ...params,
                          distance: distanceSamples,
                        })
                      }}
                      min={resolutionMs}
                      step={resolutionMs}
                      className="text-xs"
                    />
                  </div>
                  <div className="space-y-2">
                    <ParamLabel
                      name="prominence"
                      subscript="ratio"
                      tooltip="Ratio of prominence to amplitude."
                    />
                    <UnitInput
                      type="number"
                      unit="%"
                      value={Number((params.prominence_ratio * 100).toFixed(1))}
                      onChange={(e) => updateParam('prominence_ratio', parseFloat(e.target.value) / 100)}
                      onBlur={(e) => onValidateParams("detect", {
                        ...params,
                        prominence_ratio: parseFloat(e.target.value) / 100,
                      })}
                      step="0.5"
                      min="0"
                      max="100"
                      className="text-xs"
                    />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

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
