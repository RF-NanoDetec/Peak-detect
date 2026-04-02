"use client"

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { useEffect, useState } from 'react'

interface SidebarStore {
  collapsed: boolean
  setCollapsed: (collapsed: boolean) => void
  toggleCollapsed: () => void
  // Right panel state
  rightPanelCollapsed: boolean
  setRightPanelCollapsed: (collapsed: boolean) => void
  toggleRightPanel: () => void
}

const useSidebarStore = create<SidebarStore>()(
  persist(
    (set) => ({
      collapsed: false,
      setCollapsed: (collapsed) => set({ collapsed }),
      toggleCollapsed: () => set((state) => ({ collapsed: !state.collapsed })),
      rightPanelCollapsed: false,
      setRightPanelCollapsed: (rightPanelCollapsed) => set({ rightPanelCollapsed }),
      toggleRightPanel: () => set((state) => ({ rightPanelCollapsed: !state.rightPanelCollapsed })),
    }),
    {
      name: 'peak-tool-sidebar',
    }
  )
)

// Breakpoint for auto-collapse (1024px = lg breakpoint)
const COLLAPSE_BREAKPOINT = 1024

export function useSidebar() {
  const store = useSidebarStore()
  const [isSmallScreen, setIsSmallScreen] = useState(false)

  useEffect(() => {
    const checkScreenSize = () => {
      const small = window.innerWidth < COLLAPSE_BREAKPOINT
      setIsSmallScreen(small)
    }

    // Check on mount
    checkScreenSize()

    // Listen for resize
    window.addEventListener('resize', checkScreenSize)
    return () => window.removeEventListener('resize', checkScreenSize)
  }, [])

  // Auto-collapse on small screens, but allow manual override
  const effectiveCollapsed = isSmallScreen || store.collapsed

  return {
    ...store,
    isSmallScreen,
    effectiveCollapsed,
  }
}










