"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  ChevronLeft,
  ChevronRight,
  Check,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useSidebar } from "@/hooks/use-sidebar"
import { Button } from "@/components/ui/button"
import { useDataStore } from "@/lib/stores/dataStore"
import { useResultsStore } from "@/lib/stores/resultsStore"
import { advancedWorkflowSteps, workflowSteps } from "@/components/layout/workflow"

export function Sidebar() {
  const pathname = usePathname()
  const { toggleCollapsed, effectiveCollapsed } = useSidebar()
  const { resultId } = useDataStore()
  const { detectionResults } = useResultsStore()
  const hasData = Boolean(resultId)
  const hasPeaks = Boolean(detectionResults?.peak_times?.length)

  const currentStepIndex = workflowSteps.findIndex(step => pathname.startsWith(step.href))
  const getStepState = (stepId: string, index: number) => {
    if (stepId === "load") return hasData ? "Ready" : "Start"
    if (stepId === "preprocess") return hasPeaks ? "Complete" : hasData ? "Next" : "Locked"
    if (stepId === "analyze") return hasPeaks ? "Ready" : "After detection"
    if (stepId === "export") return hasPeaks ? "Ready" : "After detection"
    return currentStepIndex > index ? "Complete" : `Step ${index + 1}`
  }
  const isStepComplete = (stepId: string) => {
    if (stepId === "load") return hasData
    if (stepId === "preprocess") return hasPeaks
    if (stepId === "analyze") return hasPeaks
    return false
  }

  return (
    <aside
      className={cn(
        "border-r bg-muted/40 backdrop-blur-sm flex flex-col transition-all duration-200 ease-in-out",
        effectiveCollapsed ? "w-16" : "w-56"
      )}
    >
      {/* Header with collapse toggle */}
      <div className={cn(
        "h-14 flex items-center border-b",
        effectiveCollapsed ? "justify-center px-2" : "justify-between px-4"
      )}>
        {!effectiveCollapsed && (
          <div className="flex items-center gap-2 font-semibold overflow-hidden">
            <span className="font-display uppercase tracking-wider text-sm whitespace-nowrap text-accent">
              Workflow
            </span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleCollapsed}
          className="topbar-icon-button h-8 w-8 shrink-0 transition-transform duration-150 hover:scale-110 hover:text-foreground"
          title={effectiveCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {effectiveCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      <nav className={cn(
        "relative flex-1 py-6 space-y-2 overflow-y-auto no-scrollbar",
        effectiveCollapsed ? "px-2" : "px-4"
      )}>
        {workflowSteps.map((step, index) => {
          const Icon = step.icon
          const isActive = pathname.startsWith(step.href)
          const isCompleted = isStepComplete(step.id)

          return (
            <div key={step.id} className="relative">
              {/* Connector Line - only show when expanded */}
              {!effectiveCollapsed && index < workflowSteps.length - 1 && (
                <div className="pointer-events-none absolute left-[27px] top-[48px] bottom-[-22px] z-10 w-[4px] overflow-hidden rounded-full">
                  <div className="absolute inset-0 bg-muted-foreground/15" />
                  {(isCompleted || isActive) && (
                    <div
                      className={cn(
                        "absolute inset-0 bg-accent/20 transition-opacity duration-300",
                        isActive ? "opacity-70" : "opacity-40"
                      )}
                    />
                  )}
                </div>
              )}

              <Link
                href={step.href}
                className={cn(
                  "relative z-0 flex items-center rounded-md text-sm font-medium transition-all duration-200 group border",
                  effectiveCollapsed
                    ? "justify-center p-3"
                    : "gap-3 px-3 py-3",
                  isActive
                    ? "bg-accent/8 text-foreground border-accent/15"
                    : isCompleted
                      ? "text-foreground hover:bg-muted/60 border-border/0"
                      : "text-muted-foreground hover:bg-muted/60 hover:text-foreground border-border/0"
                )}
                title={effectiveCollapsed ? step.label : undefined}
              >
                {/* Active Step Indicator - Right Side Bar/Arrow */}
                {isActive && (
                  <div className={cn(
                    "absolute right-0 top-1/2 -translate-y-1/2 transition-all duration-200",
                    effectiveCollapsed
                      ? "w-1 h-8 rounded-l-full bg-accent/50"
                      : "w-1 h-full rounded-r-md bg-accent/50"
                  )} />
                )}
                <div className={cn(
                  "relative z-20 flex items-center justify-center h-8 w-8 rounded-full border-2 transition-all duration-200 shrink-0",
                  isActive
                    ? "border-accent bg-accent/10 text-accent"
                    : isCompleted
                      ? "border-accent bg-accent/10 text-accent"
                      : "border-muted-foreground bg-background text-muted-foreground group-hover:border-foreground group-hover:bg-muted/70 group-hover:text-foreground group-hover:scale-105"
                )}>
                  {isActive && (
                    <span className="absolute inset-[-2px] rounded-full border border-accent/15" aria-hidden />
                  )}
                  {isCompleted ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <Icon className={cn(
                      "h-4 w-4 transition-transform duration-150",
                      "group-hover:scale-110"
                    )} />
                  )}
                </div>
                {!effectiveCollapsed && (
                  <div className="flex flex-col overflow-hidden">
                    <span className={cn(
                      "leading-none truncate transition-colors duration-200",
                      isActive && "text-accent font-semibold",
                      isCompleted && "text-foreground"
                    )}>
                      {step.label}
                    </span>
                    <span className={cn(
                      "text-[10px] mt-1 font-mono uppercase tracking-widest transition-colors duration-200",
                      isActive ? "text-accent/70" :
                        isCompleted ? "text-accent/70" :
                          "text-muted-foreground"
                    )}>
                      {getStepState(step.id, index)}
                    </span>
                  </div>
                )}
              </Link>
            </div>
          )
        })}
        {!effectiveCollapsed && (
          <div className="pt-5 mt-5 border-t">
            <div className="px-1 pb-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              Advanced
            </div>
            {advancedWorkflowSteps.map((step) => {
              const Icon = step.icon
              const isActive = pathname.startsWith(step.href)
              return (
                <Link
                  key={step.id}
                  href={step.href}
                  className={cn(
                    "flex items-center gap-3 rounded-md border border-transparent px-3 py-2.5 text-sm transition-colors",
                    isActive
                      ? "bg-accent/8 text-accent border-accent/15"
                      : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                  )}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  <div className="min-w-0">
                    <div className="truncate font-medium">{step.label}</div>
                    <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                      {step.sectionLabel}
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>
        )}
      </nav>
    </aside>
  )
}
