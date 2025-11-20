"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { 
  FileUp, 
  Filter, 
  BarChart3, 
  GitBranch, 
  Download, 
  Settings 
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
    <aside className="w-60 border-r bg-card flex flex-col">
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold">Peak Analysis</h2>
        <p className="text-xs text-muted-foreground">Workflow</p>
      </div>
      
      <nav className="flex-1 p-3 space-y-6 overflow-y-auto">
        {/* Core Steps */}
        <div>
          <div className="px-3 mb-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Core
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
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span>{step.label}</span>
                </Link>
              )
            })}
          </div>
        </div>

        {/* Advanced Steps */}
        <div>
          <div className="px-3 mb-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Advanced
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
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
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
