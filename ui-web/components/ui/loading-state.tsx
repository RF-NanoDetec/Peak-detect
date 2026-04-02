import { Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface LoadingStateProps {
  text?: string
  className?: string
  spinnerClassName?: string
}

export function LoadingState({ text = "Loading...", className, spinnerClassName }: LoadingStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center p-8 space-y-4 animate-in fade-in duration-300", className)}>
      <Loader2 className={cn("h-8 w-8 animate-spin text-accent/80", spinnerClassName)} />
      {text && <p className="text-sm text-muted-foreground font-medium">{text}</p>}
    </div>
  )
}

