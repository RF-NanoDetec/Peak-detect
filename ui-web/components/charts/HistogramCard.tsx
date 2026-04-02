"use client"

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
  children 
}: { 
  active: boolean
  onClick: () => void
  children: React.ReactNode 
}) => (
  <button
    onClick={(e) => {
      e.stopPropagation()
      onClick()
    }}
    className={cn(
      "text-[9px] px-1.5 py-0.5 rounded transition-colors font-medium",
      active 
        ? "bg-primary/10 text-primary hover:bg-primary/20" 
        : "text-muted-foreground hover:bg-muted hover:text-foreground"
    )}
  >
    {children}
  </button>
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
      {/* Header - Centered Title */}
      <div className="flex items-center justify-center pt-3 pb-1 relative px-2">
        <h3 className="text-xs font-semibold text-foreground/80 uppercase tracking-wider">
          {title}
        </h3>
        
        {/* Toggles - Absolute right or floating on hover */}
        {hasData && (
          <div className="absolute right-2 top-2 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-all duration-200 bg-background/95 backdrop-blur-sm border rounded shadow-sm px-1.5 py-1 z-20">
            <div className="flex items-center gap-0.5">
              <span className="text-[9px] font-bold text-muted-foreground/70 mr-1 select-none">X</span>
              <ToggleButton active={xScale === "linear"} onClick={() => onXScaleChange("linear")}>Lin</ToggleButton>
              <ToggleButton active={xScale === "log"} onClick={() => onXScaleChange("log")}>Log</ToggleButton>
            </div>
            <div className="w-px h-3 bg-border mx-0.5" />
            <div className="flex items-center gap-0.5">
              <span className="text-[9px] font-bold text-muted-foreground/70 mr-1 select-none">Y</span>
              <ToggleButton active={yScale === "linear"} onClick={() => onYScaleChange("linear")}>Lin</ToggleButton>
              <ToggleButton active={yScale === "log"} onClick={() => onYScaleChange("log")}>Log</ToggleButton>
            </div>
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

