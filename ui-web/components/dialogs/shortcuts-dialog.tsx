"use client"

import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Keyboard } from "lucide-react"

interface ShortcutsDialogProps {
    open: boolean
    onOpenChange: (open: boolean) => void
}

export function ShortcutsDialog({ open, onOpenChange }: ShortcutsDialogProps) {
    const shortcuts = [
        { key: "Ctrl + D", description: "Toggle Dark/Light Mode" },
        { key: "?", description: "Show this help dialog" },
    ]

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <div className="flex items-center gap-2">
                        <Keyboard className="h-5 w-5 text-primary" />
                        <DialogTitle>Keyboard Shortcuts</DialogTitle>
                    </div>
                    <DialogDescription>
                        Quickly navigate and control the application.
                    </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                    <div className="grid gap-2">
                        {shortcuts.map((shortcut) => (
                            <div
                                key={shortcut.key}
                                className="flex items-center justify-between rounded-lg border p-3 bg-muted/50"
                            >
                                <span className="text-sm font-medium">{shortcut.description}</span>
                                <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-background px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100">
                                    {shortcut.key}
                                </kbd>
                            </div>
                        ))}
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    )
}
