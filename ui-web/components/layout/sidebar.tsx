"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  FileUp,
  Filter,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Check,
  GitBranch,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useSidebar } from "@/hooks/use-sidebar"
import { Button } from "@/components/ui/button"

const steps = [
  {
    id: "load",
    label: "Load Data",
    icon: FileUp,
    href: "/load",
  },
  {
    id: "preprocess",
    label: "Process & Detect",
    icon: Filter,
    href: "/preprocess",
  },
  {
    id: "analyze",
    label: "Analyze Results",
    icon: BarChart3,
    href: "/analyze",
  },
  {
    id: "double",
    label: "Double Peak",
    icon: GitBranch,
    href: "/double",
  }
]

export function Sidebar() {
  const pathname = usePathname()
  const { collapsed, toggleCollapsed, effectiveCollapsed, isSmallScreen } = useSidebar()

  // Find the current step index
  const currentStepIndex = steps.findIndex(step => pathname === step.href)

  return (
    <aside 
      className={cn(
        "border-r bg-muted/40 backdrop-blur-sm flex flex-col transition-all duration-200 ease-in-out",
        effectiveCollapsed ? "w-16" : "w-64"
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
          className="h-8 w-8 shrink-0 transition-transform duration-150 hover:scale-110 hover:text-accent"
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
        "flex-1 py-6 space-y-2 overflow-y-auto no-scrollbar",
        effectiveCollapsed ? "px-2" : "px-4"
      )}>
        {steps.map((step, index) => {
          const Icon = step.icon
          const isActive = pathname === step.href
          const isCompleted = currentStepIndex > index
          const isPending = currentStepIndex < index

          return (
            <div key={step.id} className="relative">
              {/* Connector Line - only show when expanded */}
              {!effectiveCollapsed && index < steps.length - 1 && (
                <div className={cn(
                  "absolute left-[27px] top-[48px] bottom-[-22px] w-[2px] transition-colors duration-300",
                  isCompleted ? "bg-workflow-completed/50" : 
                  isActive ? "bg-accent/30" : 
                  "bg-muted-foreground/10"
                )} />
              )}

              <Link
                href={step.href}
                className={cn(
                  "flex items-center rounded-md text-sm font-medium transition-all duration-200 group",
                  effectiveCollapsed 
                    ? "justify-center p-3" 
                    : "gap-3 px-3 py-3",
                  isActive
                    ? "bg-accent/10 text-accent shadow-sm"
                    : isCompleted
                    ? "text-foreground hover:bg-muted/50"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                )}
                title={effectiveCollapsed ? step.label : undefined}
              >
                <div className={cn(
                  "relative flex items-center justify-center h-8 w-8 rounded-full border-2 transition-all duration-200 shrink-0",
                  isActive
                    ? "border-accent bg-accent/20 text-accent animate-glow"
                    : isCompleted
                    ? "border-workflow-completed bg-workflow-completed text-white"
                    : "border-muted-foreground/20 bg-background text-muted-foreground group-hover:border-accent/50 group-hover:scale-105"
                )}>
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
                      isCompleted ? "text-workflow-completed" :
                      "text-muted-foreground"
                    )}>
                      {isCompleted ? "Completed" : `Step ${index + 1}`}
                    </span>
                  </div>
                )}
              </Link>
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
