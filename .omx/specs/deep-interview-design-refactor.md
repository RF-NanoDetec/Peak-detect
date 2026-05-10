# Spec: Professional Self-Service Design and Performance Refactor

## Metadata

- Source interview: `.omx/interviews/design-refactor-20260509T190041Z.md`
- Context snapshot: `.omx/context/design-refactor-20260509T190041Z.md`
- Final ambiguity: 18%
- Context type: brownfield

## Intent

Refactor the Peak Analysis Tool UI so future external first-time professional users can complete the full analysis path with confidence, while preserving scientific behavior and the current power-user compactness.

## Desired Outcome

The app should feel like a precise, professional scientific analysis workstation:
- compact and expert-like
- clear enough for self-service onboarding
- visibly fast with large datasets
- responsive in chart zoom/pan/rendering
- trustworthy in how it communicates data, parameters, and export readiness

## In Scope

- Review and tighten design language across layout, navigation, typography, spacing, colors, labels, empty states, loading states, and interaction states.
- Rework workflow structure if it makes first-time completion clearer.
- Improve the end-to-end path: upload -> inspect -> detect peaks -> understand results -> export detected peaks.
- Improve chart interaction and large-file responsiveness where measurement shows a bottleneck.
- Refactor UI code where it improves consistency, readability, or performance.
- Add regression and interaction tests around the preserved workflow.

## Out of Scope / Non-goals

- Do not change scientific algorithms.
- Do not change supported file formats in this phase.
- Do not add accounts, cloud storage, or a broad product/tutorial system unless a later planning pass explicitly approves it.
- Do not add new dependencies without explicit approval.

## Decision Boundaries

OMX may propose and later implement:
- navigation/workflow reordering or combining
- component layout changes
- clearer labels and embedded professional guidance
- performance measurement and optimization in the frontend
- state and component refactors that preserve behavior

OMX must not silently change:
- algorithm outputs
- file format support
- backend scientific semantics
- distribution/deployment strategy beyond planning notes

## Acceptance Criteria

- A valid first-time user path exists and is testable: upload file, inspect preview, run detection, understand result state, export detected peaks within 10 minutes.
- The UI remains compact and professional, not tutorial-heavy.
- Primary workflow state is clear at every step: no data, preview ready, loaded, filtering, detecting, detected, exportable, error.
- Chart zoom/pan/rendering remains responsive on large datasets and is measured before and after optimization.
- Large-file upload/preview path shows useful progress or state and avoids blocking the main UI where feasible.
- Existing backend/core scientific tests remain green.
- Frontend build/lint pass after changes.

## Brownfield Technical Context

- `ui-web/components/layout/sidebar.tsx`: hard-coded workflow steps and completion state.
- `ui-web/components/layout/page-shell.tsx`: split visualization/control-panel shell.
- `ui-web/app/globals.css`: global tokens, typography, chart/workflow colors, range-input styling, utility motion.
- `ui-web/components/load/LoadView.tsx`: first-run load hero, local preview, recent sessions, protocol metadata.
- `ui-web/components/preprocess/PreprocessControls.tsx`: compact filter and detection control surface.
- `ui-web/components/charts/PreprocessingChart.tsx`: data worker hydration, full-resolution toggle, chart summary, histogram state.
- `ui-web/workers/dataWorker.ts`: range slicing, min/max decimation, transfer-list messaging.
- `ui-web/components/charts/UPlotChart.tsx`: uPlot options, chart overlays, lasso, hover states, redraw behavior.

## Risks

- Over-guiding could make the app feel less professional.
- Workflow restructuring could disrupt current internal users if not backed by tests and visual QA.
- Chart performance work may regress scientific visual fidelity if decimation is changed carelessly.
- Existing uncommitted changes in the working tree must be treated as user/prior-agent work and not reverted.

