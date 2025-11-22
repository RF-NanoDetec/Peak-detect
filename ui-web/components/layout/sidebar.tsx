"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { 
  FileUp, 
  Filter, 
  BarChart3, 
  GitBranch, 
  Download, 
  Settings,
  Activity
} from "lucide-react"
import { cn } from "@/lib/utils"

const steps = [
  {
    id: "load",
    label: "Load Data",
    icon: FileUp,
    href: "/load",
    section: "core"
  },
  {
    id: "preprocess",
    label: "Process & Detect",
    icon: Filter,
    href: "/preprocess",
    section: "core"
  },
  {
    id: "analyze",
    label: "Analyze & Export",
    icon: BarChart3,
    href: "/analyze",
    section: "core"
  },
  {
    id: "double",
    label: "Double Peak",
    icon: GitBranch,
    href: "/double",
    section: "advanced"
  },
  {
    id: "export",
    label: "Export",
    icon: Download,
    href: "/export",
    section: "advanced"
  },
  {
    id: "preferences",
    label: "Preferences",
    icon: Settings,
    href: "/preferences",
    section: "advanced"
  }
]

export function Sidebar() {
  const pathname = usePathname()

  const coreSteps = steps.filter(s => s.section === "core")
  const advancedSteps = steps.filter(s => s.section === "advanced")

  return (
    <aside className="w-64 border-r bg-muted/40 backdrop-blur-sm flex flex-col">
      <div className="h-14 flex items-center px-6 border-b">
        <div className="flex items-center gap-2 font-semibold">
          {/* Icon removed as requested */}
          {/* <div className="flex items-center justify-center h-6 w-6 rounded bg-primary text-primary-foreground">
            <Activity className="h-4 w-4" />
          </div> */}
          <span>Time Trace Peak Detection</span>
        </div>
      </div>
      
      <nav className="flex-1 px-4 py-6 space-y-8 overflow-y-auto">
        {/* Core Steps */}
        <div>
          <div className="px-2 mb-3">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest">
              Workflow
            </p>
          </div>
          <div className="space-y-1">
            {coreSteps.map((step) => {
              const Icon = step.icon
              const isActive = pathname === step.href
              
              return (
                <Link
                  key={step.id}
                  href={step.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-all",
                    isActive
                      ? "bg-secondary text-secondary-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                  )}
                >
                  <Icon className={cn("h-4 w-4", isActive ? "text-primary" : "text-muted-foreground")} />
                  <span>{step.label}</span>
                </Link>
              )
            })}
          </div>
        </div>

        {/* Advanced Steps */}
        <div>
          <div className="px-2 mb-3">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest">
              Tools
            </p>
          </div>
          <div className="space-y-1">
            {advancedSteps.map((step) => {
              const Icon = step.icon
              const isActive = pathname === step.href
              
              return (
                <Link
                  key={step.id}
                  href={step.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-all",
                    isActive
                      ? "bg-secondary text-secondary-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                  )}
                >
                  <Icon className={cn("h-4 w-4", isActive ? "text-primary" : "text-muted-foreground")} />
                  <span>{step.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </nav>
    </aside>
  )
}
