import * as React from "react"
import { ChevronUp, ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Input, InputProps } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

export interface StepperInputProps extends InputProps {
    onValueChange?: (value: number) => void
    min?: number
    max?: number
    step?: number
    value?: number | string
}

export const StepperInput = React.forwardRef<HTMLInputElement, StepperInputProps>(
    ({ className, min, max, step = 1, value, onValueChange, ...props }, ref) => {
        const inputRef = React.useRef<HTMLInputElement>(null)

        const handleIncrement = () => {
            const currentValue = parseFloat(inputRef.current?.value || "0")
            const newValue = currentValue + step
            if (max !== undefined && newValue > max) return

            // Fix floating point precision issues
            const precision = step.toString().split(".")[1]?.length || 0
            const fixedValue = parseFloat(newValue.toFixed(precision))

            if (onValueChange) {
                onValueChange(fixedValue)
            }

            // Trigger native change event for compatibility
            if (inputRef.current) {
                inputRef.current.value = fixedValue.toString()
                inputRef.current.dispatchEvent(new Event('change', { bubbles: true }))
            }
        }

        const handleDecrement = () => {
            const currentValue = parseFloat(inputRef.current?.value || "0")
            const newValue = currentValue - step
            if (min !== undefined && newValue < min) return

            // Fix floating point precision issues
            const precision = step.toString().split(".")[1]?.length || 0
            const fixedValue = parseFloat(newValue.toFixed(precision))

            if (onValueChange) {
                onValueChange(fixedValue)
            }

            // Trigger native change event for compatibility
            if (inputRef.current) {
                inputRef.current.value = fixedValue.toString()
                inputRef.current.dispatchEvent(new Event('change', { bubbles: true }))
            }
        }

        // Merge refs
        React.useImperativeHandle(ref, () => inputRef.current!)

        return (
            <div className="relative flex items-center">
                <Input
                    {...props}
                    ref={inputRef}
                    type="number"
                    value={value}
                    min={min}
                    max={max}
                    step={step}
                    className={cn("pr-8", className)}
                    onChange={(e) => {
                        const val = parseFloat(e.target.value)
                        if (!isNaN(val) && onValueChange) {
                            onValueChange(val)
                        }
                        props.onChange?.(e)
                    }}
                />
                <div className="absolute right-0 flex flex-col h-full border-l border-input">
                    <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-1/2 w-6 rounded-none rounded-tr-md px-0 hover:bg-accent"
                        onClick={handleIncrement}
                        tabIndex={-1}
                    >
                        <ChevronUp className="h-3 w-3" />
                    </Button>
                    <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-1/2 w-6 rounded-none rounded-br-md px-0 border-t border-input hover:bg-accent"
                        onClick={handleDecrement}
                        tabIndex={-1}
                    >
                        <ChevronDown className="h-3 w-3" />
                    </Button>
                </div>
            </div>
        )
    }
)
StepperInput.displayName = "StepperInput"
