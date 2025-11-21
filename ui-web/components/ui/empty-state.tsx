import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface EmptyStateProps {
  icon?: LucideIcon
  title: string
  description?: string
  action?: React.ReactNode
  className?: string
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center p-8 text-center animate-in fade-in zoom-in-95 duration-300", className)}>
      {Icon && (
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted/50 mb-4 ring-1 ring-border">
          <Icon className="h-8 w-8 text-muted-foreground/80" />
        </div>
      )}
      <h3 className="text-lg font-medium tracking-tight">{title}</h3>
      {description && (
        <p className="text-sm text-muted-foreground mt-2 max-w-sm mx-auto text-balance">
          {description}
        </p>
      )}
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}

