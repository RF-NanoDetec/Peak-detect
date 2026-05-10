import { BarChart3, Download, FileUp, Filter, GitBranch } from "lucide-react"

export const workflowSteps = [
  {
    id: "load",
    label: "Load Data",
    description: "Import measurement files and inspect the preview",
    icon: FileUp,
    href: "/load",
  },
  {
    id: "preprocess",
    label: "Process & Detect",
    description: "Filter the signal and detect peaks",
    icon: Filter,
    href: "/preprocess",
  },
  {
    id: "analyze",
    label: "Analyze Results",
    description: "Review detected peak distributions and trends",
    icon: BarChart3,
    href: "/analyze",
  },
  {
    id: "export",
    label: "Export Peaks",
    description: "Download detected peak data",
    icon: Download,
    href: "/export",
  },
] as const

export const advancedWorkflowSteps = [
  {
    id: "double",
    label: "Double Peak",
    description: "Analyze consecutive peak pairs",
    sectionLabel: "Advanced",
    icon: GitBranch,
    href: "/double",
  },
] as const
