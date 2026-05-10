# Plan: Design and Performance Refactor

## Requirements Summary

Refactor `ui-web` toward a compact professional scientific analysis workstation for future external first-time users. Preserve algorithms and supported file formats. The end-to-end workflow must make it easy to upload, inspect, detect peaks, understand results, and export detected peaks in one sitting.

## Design Direction

Use a restrained "scientific workstation" language:
- data-first chart surfaces
- compact controls with precise labels
- embedded state communication instead of tutorial copy
- consistent tokens for workflow, chart, warning, success, and export readiness
- motion only where it clarifies response, progress, or state

This should feel professional and self-service, not marketing-style SaaS and not a dense hardware-instrument clone.

## Current Evidence

- The sidebar defines a fixed workflow sequence in `ui-web/components/layout/sidebar.tsx:18`.
- The shared split layout already supports visualization plus controls in `ui-web/components/layout/page-shell.tsx:14` and `ui-web/components/layout/page-shell.tsx:28`.
- The load page has a first-run entry point and compact three-step explainer in `ui-web/components/load/LoadView.tsx:83`.
- Local preview already reports rows, time range, amplitude, and parse duration in `ui-web/components/load/LoadView.tsx:196`.
- Protocol metadata is always present under the load view in `ui-web/components/load/LoadView.tsx:261`.
- Preprocess controls expose professional parameters compactly in `ui-web/components/preprocess/PreprocessControls.tsx:49`.
- Chart rendering already uses worker-fed displayed data in `ui-web/components/charts/PreprocessingChart.tsx:160`.
- Dynamic downsampling targets roughly 1000-2500 points based on width in `ui-web/components/charts/PreprocessingChart.tsx:195`.
- Zoom range requests are debounced at 50ms in `ui-web/components/charts/PreprocessingChart.tsx:534`.
- The data worker uses min/max decimation and transfer-list responses in `ui-web/workers/dataWorker.ts:259` and `ui-web/workers/dataWorker.ts:393`.
- `UPlotChart` centralizes scale handling, overlays, lasso, and redraw behavior in `ui-web/components/charts/UPlotChart.tsx:245`.
- Global design tokens and motion utilities live in `ui-web/app/globals.css:8`.

## Cleanup Plan Before Code

Behavior lock first:
- Run existing backend/core tests to prove scientific behavior is unchanged.
- Run `npm run build` and available frontend lint/type checks to establish current frontend baseline.
- Add or identify a focused UI regression path for load -> preview -> detect -> export before restructuring workflow.

Smell-focused passes:
1. State and copy audit: identify inconsistent labels, unclear states, redundant explanatory text, and missing export readiness cues.
2. Layout consistency pass: normalize page headers, control panels, chart info bars, empty states, and action placement.
3. Workflow communication pass: improve first-time path without adding a tutorial layer.
4. Performance measurement pass: add repeatable measurements for local preview, full-res chart load, zoom/pan responsiveness, and parameter-change feedback.
5. Targeted performance pass: optimize only measured bottlenecks.

## Implementation Steps

1. Baseline audit and measurements
   - Inventory all route states for `/load`, `/preprocess`, `/analyze`, `/double`, `/export`, `/preferences`.
   - Capture screenshots for empty, loaded, detecting, detected, and exportable states.
   - Record large-file timing for local preview, upload, chart hydration, full-resolution toggle, zoom/pan, and export readiness.
   - Keep existing algorithms and formats untouched.

2. Define a small design contract
   - Consolidate the visual rules in `ui-web/app/globals.css` and existing UI primitives.
   - Define tokens/usages for workflow state, chart overlays, action hierarchy, warning/error/success, and disabled/export-ready states.
   - Avoid adding a new design-system layer unless repeated code proves it is needed.

3. Refactor workflow communication
   - Replace the load-page mini explainer with a compact professional workflow status surface that reflects actual app state.
   - Make the current sidebar completion model more truthful than `canComplete` plus route index if persisted state is available.
   - Consider merging or visually connecting Analyze/Export if export is the natural final step after peak detection.
   - Keep `Double Peak` advanced rather than blocking the main first-time path.

4. Improve first-time professional guidance
   - Add concise embedded cues where users need confidence: file requirements, preview meaning, detection parameter intent, peak count/result quality, and export readiness.
   - Keep labels precise and compact; use tooltips and state text, not long tutorial panels.
   - Make empty/error/loading states explain the next valid action.

5. Tighten chart interaction and performance
   - Preserve the worker/downsampling design in `PreprocessingChart` and `dataWorker`.
   - Measure whether `UPlotChart` redraw hooks, lasso loops, hover state, or width segment rendering cause frame drops on large datasets.
   - Avoid changing decimation semantics until visual fidelity tests confirm peak visibility is preserved.
   - Add a visible but compact indicator for displayed points, full-res mode, decimation, and zoom range.

6. Simplify component boundaries
   - Keep layout primitives in `components/layout`.
   - Keep generic primitives in `components/ui`.
   - Keep route-specific professional guidance inside route components.
   - Split only when a component has multiple responsibilities and tests/screenshots show the refactor is safe.

7. Verify end-to-end
   - Run backend/core tests.
   - Run frontend build/lint.
   - Use browser verification on desktop and narrow viewports.
   - Confirm the first-time path: select file, inspect preview, load, detect peaks, inspect results, export detected peaks.
   - Confirm chart pan/zoom remains responsive with a large dataset.

## Acceptance Criteria

- First-time professional path is possible without outside instruction: upload -> inspect -> detect -> understand -> export.
- Main workflow can be completed in under 10 minutes by a user with a valid supported file.
- Scientific algorithms and supported file formats are unchanged.
- UI remains compact and expert-like.
- Every major route has clear empty, loading, error, and ready states.
- Chart interaction remains smooth on large datasets, with before/after measurement.
- No new dependency is introduced.

## Risks and Mitigations

- Risk: making the app too tutorial-like.
  - Mitigation: embed short state/action cues and preserve dense professional controls.

- Risk: workflow restructuring confuses current users.
  - Mitigation: preserve route names where possible and verify old workflow path still works.

- Risk: performance changes alter visual interpretation.
  - Mitigation: test decimated/full-resolution visual behavior before and after; do not alter algorithms.

- Risk: current dirty working tree contains unrelated work.
  - Mitigation: review diffs before editing and do not revert unrelated changes.

## Verification Steps

- `pytest`
- `cd ui-web; npm run build`
- `cd ui-web; npm run lint` if the Next lint script works in this Next version
- Manual/browser QA for `/load`, `/preprocess`, `/analyze`, `/double`, and `/export`
- Large-dataset performance notes for preview, full-res toggle, zoom/pan, and export readiness

## ADR

Decision:
- Evolve the app into a compact professional self-service scientific workstation.

Drivers:
- First-time external users need to complete the full workflow without internal onboarding.
- Existing professional users need compact controls and high performance.
- Smooth large-dataset interaction is a core trust signal.

Alternatives considered:
- Generic SaaS dashboard: rejected because it would dilute the scientific workstation feel.
- Dense lab-instrument clone: rejected because onboarding is already hard.
- Tutorial-first beginner product: rejected because the users are professionals and need embedded clarity, not hand-holding.

Why chosen:
- The workstation direction preserves professional density while making workflow state, labels, and next actions clearer.

Consequences:
- Navigation and information architecture may change.
- Algorithms and file formats stay fixed.
- Performance work must be measurement-led.

Follow-ups:
- A later plan can address public distribution/web deployment and richer file-format support as separate product decisions.
