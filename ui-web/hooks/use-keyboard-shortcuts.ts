"use client"

import { useEffect } from "react"
import { useTheme } from "@/hooks/use-theme"

interface UseKeyboardShortcutsProps {
    onOpenShortcuts: () => void
}

export function useKeyboardShortcuts({ onOpenShortcuts }: UseKeyboardShortcutsProps) {
    const { theme, setTheme } = useTheme()

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Ignore if input or textarea is focused
            if (
                document.activeElement instanceof HTMLInputElement ||
                document.activeElement instanceof HTMLTextAreaElement
            ) {
                return
            }

            // Toggle Theme: Ctrl + D
            if (e.ctrlKey && e.key === "d") {
                e.preventDefault()
                setTheme(theme === "dark" ? "light" : "dark")
            }

            // Show Shortcuts: ? (Shift + /)
            if (e.key === "?") {
                e.preventDefault()
                onOpenShortcuts()
            }
        }

        window.addEventListener("keydown", handleKeyDown)
        return () => window.removeEventListener("keydown", handleKeyDown)
    }, [theme, setTheme, onOpenShortcuts])
}
