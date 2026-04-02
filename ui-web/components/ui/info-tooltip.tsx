"use client"

import { Info } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface InfoTooltipProps {
  content: string
}

export function InfoTooltip({ content }: InfoTooltipProps) {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={300}>
        <TooltipTrigger asChild>
          <Info className="h-3.5 w-3.5 text-muted-foreground/70 hover:text-accent cursor-help transition-colors inline-block ml-1.5 translate-y-[2px]" />
        </TooltipTrigger>
        <TooltipContent className="max-w-[300px] text-xs leading-relaxed">
          <p>{content}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

