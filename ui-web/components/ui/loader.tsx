"use client"

import { cn } from "@/lib/utils"

interface LoaderProps {
    className?: string
    size?: "sm" | "md" | "lg"
    text?: string
}

export function Loader({ className, size = "md", text }: LoaderProps) {
    const sizeClasses = {
        sm: "w-8 h-8",
        md: "w-12 h-12",
        lg: "w-16 h-16",
    }

    return (
        <div className={cn("flex flex-col items-center justify-center gap-3", className)}>
            <div className={cn("relative", sizeClasses[size])}>
                <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-accent animate-draw"
                    style={{
                        strokeDasharray: 100,
                    }}
                >
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
            </div>
            {text && (
                <p className="text-sm text-muted-foreground animate-pulse font-medium">
                    {text}
                </p>
            )}
        </div>
    )
}
