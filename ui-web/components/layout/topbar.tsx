"use client"

import { useState } from "react"
import { Moon, Sun, Download, Settings, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/hooks/use-theme"
import { useKeyboardShortcuts } from "@/hooks/use-keyboard-shortcuts"
import { ShortcutsDialog } from "@/components/dialogs/shortcuts-dialog"
import Image from "next/image"

export function Topbar() {
  const { theme, toggleTheme } = useTheme()
  const [showShortcuts, setShowShortcuts] = useState(false)

  useKeyboardShortcuts({
    onOpenShortcuts: () => setShowShortcuts(true),
  })

  return (
    <header className="h-14 border-b bg-card flex items-center justify-between px-4">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <Image
            src={theme === 'dark' ? '/logo-dark.svg' : '/logo-light.svg'}
            alt="Peak Analysis Tool Logo"
            width={70}
            height={53}
            className="h-6 w-auto"
            priority
          />
          <h1 className="text-base font-semibold leading-tight">Peak Analysis Tool</h1>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          title="Export Data"
          asChild
        >
          <a href="/export">
            <Download className="h-5 w-5 text-muted-foreground" />
          </a>
        </Button>
        <Button
          variant="ghost"
          size="icon"
          title="Settings"
          asChild
        >
          <a href="/preferences">
            <Settings className="h-5 w-5 text-muted-foreground" />
          </a>
        </Button>
        <Button
          variant="ghost"
          size="icon"
          title="Keyboard Shortcuts (?)"
          onClick={() => setShowShortcuts(true)}
        >
          <HelpCircle className="h-5 w-5 text-muted-foreground" />
        </Button>
        <div className="w-px h-6 bg-border mx-1" />
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode (Ctrl+D)`}
        >
          {theme === 'dark' ? (
            <Sun className="h-5 w-5" />
          ) : (
            <Moon className="h-5 w-5" />
          )}
        </Button>
      </div>

      <ShortcutsDialog
        open={showShortcuts}
        onOpenChange={setShowShortcuts}
      />
    </header>
  )
}
