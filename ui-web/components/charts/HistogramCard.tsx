"use client"

import type { ReactNode } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { UPlotHistogram } from "./UPlotHistogram"
import { cn } from "@/lib/utils"

export type AxisScaleOption = "linear" | "log"

interface HistogramCardProps {
  title: string
  data: { bins: number[]; counts: number[]; bin_edges?: number[] }
  color: string
  xLabel: string
  yLabel: string
  xScale: AxisScaleOption
  yScale: AxisScaleOption
  onXScaleChange: (value: AxisScaleOption) => void
  onYScaleChange: (value: AxisScaleOption) => void
  verticalLines?: Array<{ value: number; color?: string; label?: string }>
  className?: string
  loading?: boolean
}

const ToggleButton = ({ 
  active, 
  onClick, 
  children,
  title,
}: { 
  active: boolean
  onClick: () => void
  children: ReactNode
  title: string
}) => (
  <button
    type="button"
    aria-pressed={active}
    title={title}
    onClick={(e) => {
      e.stopPropagation()
      onClick()
    }}
    className={cn(
      "h-5 min-w-7 rounded-md border px-1.5 text-[9px] font-medium leading-none transition-colors",
      "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
      active 
        ? "border-accent bg-accent text-accent-foreground shadow-sm" 
        : "border-border/50 text-muted-foreground hover:border-accent/60 hover:bg-accent/10 hover:text-foreground"
    )}
  >
    {children}
  </button>
)

const ScaleControl = ({
  axis,
  value,
  onChange,
}: {
  axis: "X" | "Y"
  value: AxisScaleOption
  onChange: (value: AxisScaleOption) => void
}) => (
  <div className="flex items-center gap-1 rounded-md bg-muted/20 px-1 py-0.5">
    <span className="font-mono text-[9px] font-semibold leading-none text-muted-foreground">
      {axis}
    </span>
    <div
      className="inline-flex items-center gap-0.5 text-muted-foreground"
      role="group"
      aria-label={`${axis} axis scale`}
    >
      <ToggleButton
        active={value === "linear"}
        onClick={() => onChange("linear")}
        title={`${axis} axis linear scale`}
      >
        Lin
      </ToggleButton>
      <ToggleButton
        active={value === "log"}
        onClick={() => onChange("log")}
        title={`${axis} axis log scale`}
      >
        Log
      </ToggleButton>
    </div>
  </div>
)

export function HistogramCard({
  title,
  data,
  color,
  xLabel,
  yLabel,
  xScale,
  yScale,
  onXScaleChange,
  onYScaleChange,
  verticalLines,
  className,
  loading
}: HistogramCardProps) {
  const hasData = data?.bins?.length > 0

  return (
    <Card className={cn("border bg-card shadow-sm overflow-hidden relative group", className)}>
      {/* Header */}
      <div className="flex min-h-8 items-start justify-between gap-2 px-2 pb-1 pt-2.5">
        <h3 className="min-w-0 truncate pt-1 text-xs font-semibold text-foreground/80 uppercase tracking-wider">
          {title}
        </h3>
        
        {/* Compact scale controls */}
        {hasData && (
          <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
            <ScaleControl axis="X" value={xScale} onChange={onXScaleChange} />
            <ScaleControl axis="Y" value={yScale} onChange={onYScaleChange} />
          </div>
        )}
      </div>

      <CardContent className="p-0">
        {loading ? (
           <div className="h-[200px] flex items-center justify-center text-xs text-muted-foreground">
             Loading...
           </div>
        ) : hasData ? (
          <div className="p-2">
            <UPlotHistogram
              data={data}
              xLabel={xLabel}
              yLabel={yLabel}
              color={color}
              xScaleType={xScale}
              yScaleType={yScale}
              height={200}
              className="w-full"
              verticalLines={verticalLines}
            />
          </div>
        ) : (
          <div className="h-[200px] flex items-center justify-center text-xs text-muted-foreground">
            No {title.toLowerCase()} data
          </div>
        )}
      </CardContent>
    </Card>
  )
}

