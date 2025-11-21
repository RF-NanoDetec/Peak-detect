"use client"

import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/hooks/use-theme"
import Image from "next/image"

export function Topbar() {
  const { theme, toggleTheme } = useTheme()

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
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? (
            <Sun className="h-5 w-5" />
          ) : (
            <Moon className="h-5 w-5" />
          )}
        </Button>
      </div>
    </header>
  )
}
