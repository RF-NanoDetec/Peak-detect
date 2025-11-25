"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  FileUp,
  Filter,
  BarChart3,
} from "lucide-react"
import { cn } from "@/lib/utils"

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
    icon: Filter,
    href: "/double",
  }
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 border-r bg-muted/40 backdrop-blur-sm flex flex-col">
      <div className="h-14 flex items-center px-6 border-b">
        <div className="flex items-center gap-2 font-semibold">
          <span>Workflow Progress</span>
        </div>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto no-scrollbar">
        {steps.map((step, index) => {
          const Icon = step.icon
          const isActive = pathname === step.href

          return (
            <div key={step.id} className="relative">
              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div className={cn(
                  "absolute left-[27px] top-[48px] bottom-[-22px] w-[2px]",
                  isActive ? "bg-primary/20" : "bg-muted-foreground/10"
                )} />
              )}

              <Link
                href={step.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-3 rounded-md text-sm font-medium transition-all group",
                  isActive
                    ? "bg-secondary text-secondary-foreground shadow-sm"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                )}
              >
                <div className={cn(
                  "flex items-center justify-center h-8 w-8 rounded-full border-2 transition-colors",
                  isActive
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-muted-foreground/20 bg-background text-muted-foreground group-hover:border-primary/50"
                )}>
                  <Icon className="h-4 w-4" />
                </div>
                <div className="flex flex-col">
                  <span className={cn("leading-none", isActive && "text-primary")}>{step.label}</span>
                  <span className="text-[10px] text-muted-foreground mt-1 font-normal">Step {index + 1}</span>
                </div>
              </Link>
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
