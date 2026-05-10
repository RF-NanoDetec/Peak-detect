# Deep Interview: Design and Performance Refactor

Metadata:
- Profile: standard
- Context type: brownfield
- Final ambiguity: 18%
- Threshold: 20%
- Context snapshot: `.omx/context/design-refactor-20260509T190041Z.md`

## Clarified Direction

The app should become a compact professional scientific analysis workstation for future external first-time users, while still serving the current research group. It should not feel like a generic SaaS dashboard or a tutorial-heavy beginner app. The product should feel fast, precise, data-first, and trustworthy.

## Key Answers

1. Product feeling:
   - Do not force a single rigid style category.
   - Position the app for professional scientific users who may not know the internal group workflow.

2. Audience:
   - Optimize for future external first-time users.
   - This also helps current internal onboarding because current usage is limited and onboarding is hard.

3. Non-goals:
   - Do not change scientific algorithms.
   - Do not change supported file formats for this phase.
   - Larger product additions may be discussed but must not be assumed.

4. Decision boundaries:
   - Workflow steps may move or combine if that makes the end-to-end path clearer.
   - Navigation and information architecture are in scope.

5. Success criteria:
   - A first-time external user with a valid time-series file should be able to upload, inspect, detect peaks, understand the result enough to proceed, and export detected peaks within 10 minutes.

6. Guidance style:
   - Keep the UI compact and expert-like.
   - Use layout, labels, states, defaults, and design communication instead of a full tutorial layer.

7. Performance priority:
   - Prioritize large-file upload/use speed and responsive chart zoom/pan/rendering.
   - Smooth big-dataset interaction is a trust signal.

## Pressure-Pass Finding

The assumption that "first-time user" means "beginner tutorial product" was rejected. The correct interpretation is professional self-service: users should not need hand-holding, but the interface should communicate the path, state, and consequences clearly.

## Brownfield Evidence

- Workflow navigation is hard-coded in `ui-web/components/layout/sidebar.tsx`.
- Shared layout uses `PageShell`, `PageVisualization`, and `PageControls` in `ui-web/components/layout/page-shell.tsx`.
- Load page already previews selected data and exposes recent sessions in `ui-web/components/load/LoadView.tsx`.
- Preprocess controls are compact and parameter-heavy in `ui-web/components/preprocess/PreprocessControls.tsx`.
- Charting already uses a worker and downsampling architecture in `ui-web/components/charts/PreprocessingChart.tsx` and `ui-web/workers/dataWorker.ts`.
- Chart rendering and overlays are centralized in `ui-web/components/charts/UPlotChart.tsx`.

