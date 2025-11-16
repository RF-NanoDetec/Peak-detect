import * as React from "react"
import { cn } from "@/lib/utils"

interface PageShellProps {
  children: React.ReactNode
  className?: string
}

export function PageShell({ children, className }: PageShellProps) {
  return (
    <div className={cn("flex-1 flex overflow-hidden min-h-0 min-w-0 h-full", className)}>
      {children}
    </div>
  )
}

interface PageControlsProps {
  children: React.ReactNode
  className?: string
}

export function PageControls({ children, className }: PageControlsProps) {
  return (
    <div
      className={cn(
        "basis-[420px] max-w-full shrink-0 border-r bg-background p-4 space-y-4 overflow-y-auto flex flex-col min-h-0 min-w-0",
        className
      )}
    >
      {children}
    </div>
  )
}

interface PageVisualizationProps {
  children: React.ReactNode
  className?: string
}

export function PageVisualization({ children, className }: PageVisualizationProps) {
  return (
    <div className={cn("flex-1 flex flex-col bg-background min-h-0 min-w-0 overflow-auto", className)}>
      {children}
    </div>
  )
}

