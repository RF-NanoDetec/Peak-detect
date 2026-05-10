import { Input, type InputProps } from "@/components/ui/input"
import { StepperInput, type StepperInputProps } from "@/components/ui/stepper-input"
import { cn } from "@/lib/utils"

export function UnitInput({
  unit,
  className,
  unitClassName,
  ...props
}: InputProps & { unit: string; unitClassName?: string }) {
  return (
    <div className="relative">
      <Input
        {...props}
        className={cn(unit.length > 2 ? "pr-16" : "pr-10", className)}
      />
      <span
        className={cn(
          "pointer-events-none absolute inset-y-0 right-3 flex items-center text-[10px] text-muted-foreground",
          unitClassName
        )}
      >
        {unit}
      </span>
    </div>
  )
}

export function UnitStepperInput({
  unit,
  className,
  unitClassName,
  ...props
}: StepperInputProps & { unit: string; unitClassName?: string }) {
  return (
    <div className="relative">
      <StepperInput
        {...props}
        className={cn(unit.length > 2 ? "pr-20" : "pr-14", className)}
      />
      <span
        className={cn(
          "pointer-events-none absolute inset-y-0 right-8 flex items-center text-[10px] text-muted-foreground",
          unitClassName
        )}
      >
        {unit}
      </span>
    </div>
  )
}
