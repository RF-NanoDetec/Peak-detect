"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { ChevronLeft, ChevronRight, PanelRightClose, PanelRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useSidebar } from "@/hooks/use-sidebar"

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
  widthClassName?: string
}

export function PageControls({ children, className, widthClassName }: PageControlsProps) {
  const { rightPanelCollapsed, toggleRightPanel } = useSidebar()
  const panelWidthClasses = widthClassName || "w-[280px] lg:w-[300px] xl:w-[320px]"

  return (
    <>
      {/* Collapsed state - just show toggle button */}
      {rightPanelCollapsed && (
        <div className="shrink-0 border-l bg-muted/20 flex flex-col items-center py-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleRightPanel}
            className="h-8 w-8"
            title="Show controls panel"
          >
            <PanelRight className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Expanded state */}
      {!rightPanelCollapsed && (
        <div
          className={cn(
            "shrink-0 border-l bg-muted/20 flex flex-col min-h-0 min-w-0 transition-all duration-200",
            // Responsive widths: narrower on smaller screens
            panelWidthClasses,
            className
          )}
        >
          {/* Toggle button */}
          <div className="flex items-center justify-end px-2 py-2 border-b">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleRightPanel}
              className="h-7 w-7"
              title="Hide controls panel"
            >
              <PanelRightClose className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Scrollable content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {children}
          </div>
        </div>
      )}
    </>
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
