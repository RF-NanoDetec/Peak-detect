# Context Snapshot: Design and Performance Refactor

Task statement:
- Review the current UI design language and interaction model, then interview the user before creating a refactor plan for a faster, nicer, more consistent application.

Desired outcome:
- A concrete plan for improving visual consistency, responsiveness, interaction quality, and performance without changing scientific behavior unintentionally.

Stated solution:
- Start with a design/code review and user interview, then create a refactor plan.

Probable intent hypothesis:
- The user wants the product to feel more polished and professional while preserving the existing peak-analysis workflow and avoiding a broad rewrite.

Known facts/evidence:
- Frontend is in `ui-web/`, built with Next.js 14, TypeScript, Tailwind, Radix UI primitives, lucide icons, Zustand stores, and uPlot charting.
- Main workflow pages are `load`, `preprocess`, `analyze`, `double`, `export`, and `preferences`.
- Shared layout is in `ui-web/components/layout/` with a topbar, sidebar workflow navigation, and split page shell/control panel pattern.
- Shared design tokens live mainly in `ui-web/app/globals.css`.
- Chart-heavy interaction surfaces live in `ui-web/components/charts/`, including `UPlotChart`, `UPlotHistogram`, `PreprocessingChart`, and `AnalyzeTimeSeries`.
- Existing tests cover backend/core behavior; frontend package exposes `dev`, `build`, `start`, and `lint` scripts.
- Working tree already has user or prior-agent modifications in multiple files including `service/handlers.py`, several `ui-web` components, `.omx` state files, and untracked chart/log files.

Constraints:
- No source edits during deep-interview mode.
- Cleanup/refactor work should have a cleanup plan before code and should lock behavior with regression tests where behavior is not already protected.
- No new dependencies unless explicitly requested.
- Keep diffs small, reviewable, and reversible.

Unknowns/open questions:
- Which product feel should dominate: lab instrument/operator console, SaaS analytics dashboard, or polished scientific desktop app.
- Whether the priority is visual consistency, chart interaction speed, page/navigation ergonomics, or reducing code complexity first.
- What “fast” means in measurable terms: load time, chart pan/zoom latency, parameter-change latency, upload/preview time, route transitions, or build/runtime performance.
- Which current screens feel weakest to the user.

Decision-boundary unknowns:
- How far OMX may change visual language without confirmation.
- Whether workflow/navigation structure may change or only styling/components.
- Whether interaction affordances may change if scientific outputs remain the same.
- Whether performance work may touch backend/API boundaries or should stay frontend-only.

Likely codebase touchpoints:
- `ui-web/app/globals.css`
- `ui-web/components/layout/*`
- `ui-web/components/ui/*`
- `ui-web/components/load/*`
- `ui-web/components/preprocess/*`
- `ui-web/components/analyze/*`
- `ui-web/components/double/*`
- `ui-web/components/charts/*`
- `ui-web/lib/stores/*`
- `ui-web/hooks/*`
