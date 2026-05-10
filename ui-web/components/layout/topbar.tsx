"use client"

import { useState } from "react"
import Link from "next/link"
import { Moon, Sun, Settings, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/hooks/use-theme"
import { useKeyboardShortcuts } from "@/hooks/use-keyboard-shortcuts"
import { ShortcutsDialog } from "@/components/dialogs/shortcuts-dialog"
import Image from "next/image"
import { cn } from "@/lib/utils"
import { workflowSteps } from "@/components/layout/workflow"

const exportStep = workflowSteps.find((step) => step.id === "export")!

export function Topbar() {
  const { theme, toggleTheme } = useTheme()
  const [showShortcuts, setShowShortcuts] = useState(false)

  useKeyboardShortcuts({
    onOpenShortcuts: () => setShowShortcuts(true),
  })

  return (
    <header className="h-14 border-b bg-card flex items-center justify-between px-4">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3 group">
          <Image
            src={theme === 'dark' ? '/logo-dark.svg' : '/logo-light.svg'}
            alt="Peak Analysis Tool Logo"
            width={70}
            height={53}
            className="h-6 w-auto transition-transform duration-200 group-hover:scale-105"
            priority
          />
          <h1 className="text-base font-semibold leading-tight hidden sm:block transition-colors duration-200 group-hover:text-accent">
            Peak Analysis Tool
          </h1>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          title={exportStep.label}
          asChild
          className="group topbar-icon-button"
        >
          <Link href={exportStep.href}>
            <exportStep.icon className={cn(
              "h-5 w-5 text-muted-foreground",
              "transition-all duration-150",
              "group-hover:text-foreground group-hover:scale-110"
            )} />
          </Link>
        </Button>
        <Button
          variant="ghost"
          size="icon"
          title="Settings"
          asChild
          className="group topbar-icon-button"
        >
          <Link href="/preferences">
            <Settings className={cn(
              "h-5 w-5 text-muted-foreground",
              "transition-all duration-150",
              "group-hover:text-foreground group-hover:scale-110 group-hover:rotate-45"
            )} />
          </Link>
        </Button>
        <Button
          variant="ghost"
          size="icon"
          title="Keyboard Shortcuts (?)"
          onClick={() => setShowShortcuts(true)}
          className="group topbar-icon-button"
        >
          <HelpCircle className={cn(
            "h-5 w-5 text-muted-foreground",
            "transition-all duration-150",
            "group-hover:text-foreground group-hover:scale-110"
          )} />
        </Button>
        <div className="w-px h-6 bg-border mx-1" />
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode (Ctrl+D)`}
          className="group topbar-icon-button"
        >
          {theme === 'dark' ? (
            <Sun className={cn(
              "h-5 w-5",
              "transition-all duration-200",
              "group-hover:text-foreground group-hover:scale-110 group-hover:rotate-180"
            )} />
          ) : (
            <Moon className={cn(
              "h-5 w-5",
              "transition-all duration-200",
              "group-hover:text-foreground group-hover:scale-110 group-hover:-rotate-12"
            )} />
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
