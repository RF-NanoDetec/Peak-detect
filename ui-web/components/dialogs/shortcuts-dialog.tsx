"use client"

import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Keyboard, FileText, BookOpen } from "lucide-react"

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
                        <Keyboard className="h-5 w-5 text-accent" />
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

                    <div className="border-t pt-4">
                        <h4 className="text-sm font-medium mb-3">Documentation</h4>
                        <div className="grid gap-2">
                            <a 
                                href="/docs/mathematical_reference.pdf" 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="flex items-center justify-between rounded-lg border p-3 hover:bg-muted/50 transition-colors group"
                            >
                                <div className="flex items-center gap-2">
                                    <FileText className="h-4 w-4 text-muted-foreground group-hover:text-accent" />
                                    <span className="text-sm font-medium">Mathematical Reference</span>
                                </div>
                                <span className="text-xs text-muted-foreground">PDF</span>
                            </a>
                            <a 
                                href="/docs/user_manual.pdf" 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="flex items-center justify-between rounded-lg border p-3 hover:bg-muted/50 transition-colors group"
                            >
                                <div className="flex items-center gap-2">
                                    <BookOpen className="h-4 w-4 text-muted-foreground group-hover:text-accent" />
                                    <span className="text-sm font-medium">User Manual</span>
                                </div>
                                <span className="text-xs text-muted-foreground">PDF</span>
                            </a>
                        </div>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    )
}
