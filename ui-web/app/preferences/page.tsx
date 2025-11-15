"use client"

import { Settings } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { PageShell, PageControls, PageVisualization } from "@/components/layout/page-shell"
import { useTheme } from "@/hooks/use-theme"

export default function PreferencesPage() {
  const { theme, toggleTheme } = useTheme()

  return (
    <PageShell>
      <PageControls>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">Preferences</h2>
            <p className="text-sm text-muted-foreground">
              Customize application settings
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Appearance</CardTitle>
              <CardDescription>
                Customize the look and feel
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Theme</label>
                <select 
                  className="w-full px-3 py-2 rounded-md border bg-background text-sm"
                  value={theme}
                  onChange={(e) => {
                    if (e.target.value !== theme) {
                      toggleTheme()
                    }
                  }}
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Density</label>
                <select className="w-full px-3 py-2 rounded-md border bg-background text-sm">
                  <option>Comfortable</option>
                  <option>Compact</option>
                </select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Defaults</CardTitle>
              <CardDescription>
                Set default parameter values
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Time Resolution</label>
                <input
                  type="number"
                  className="w-full px-3 py-2 rounded-md border bg-background text-sm"
                  placeholder="0.0001"
                  defaultValue="0.0001"
                  step="0.0001"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Recent Files Limit</label>
                <input
                  type="number"
                  className="w-full px-3 py-2 rounded-md border bg-background text-sm"
                  placeholder="10"
                  defaultValue="10"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">About</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Version:</span>
                <span className="font-medium">1.0.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Backend:</span>
                <span className="font-medium">Online</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </PageControls>

      <PageVisualization>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center space-y-4 max-w-md">
            <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
              <Settings className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Application Settings</h3>
              <p className="text-sm text-muted-foreground mt-2">
                Configure your preferences from the left panel
              </p>
            </div>
          </div>
        </div>
      </PageVisualization>
    </PageShell>
  )
}

