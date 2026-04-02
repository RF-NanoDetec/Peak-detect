# Peak Analysis Tool

![Peak Analysis Tool interface](docs/figures/Peak_analysis.png)

> **Repository status:** private/internal-use documentation and development workspace.
> The repository is not offered as open-source software and is being kept in an all-rights-reserved posture while future commercial licensing terms are defined.

Peak Analysis Tool is a browser-based application for loading detector traces, preprocessing noisy time-series data, detecting peaks, inspecting distributions, running double-peak analysis, and exporting analysis-ready results.

## Intended Audience

This repository is currently maintained for:
- internal research and development
- controlled collaborator review
- future commercial product preparation

It is **not** presented as a public contribution or reuse-ready open-source project.

## What the application does today

The current web application supports a fixed workflow:
1. **Load Data** - choose files with the browser file picker or restore a recent session
2. **Process & Detect** - apply optional filtering and run peak detection
3. **Analyze Results** - inspect detected peaks, metrics, and derived plots
4. **Double Peak** - analyze consecutive peak pairs

**Export** is available as a separate page/action after data has been loaded and analyzed.

Core capabilities currently implemented:
- local upload of `.txt`, `.xls`, and `.xlsx` input files
- protocol metadata capture during loading
- photon-counter dead-time correction during load
- Butterworth and Savitzky-Golay preprocessing
- peak detection with prominence, distance, width, and ratio controls
- time-series, histogram, and double-peak analysis views
- export of single-peak, double-peak, and combined outputs

## Supported input data

The current loader behavior is intentionally narrow and should be documented as such:
- supported files: `.txt`, `.xls`, `.xlsx`
- the loader reads the **first two columns only**
- `.txt` files are parsed as **tab-delimited** text
- blank lines and `#` comments are ignored in `.txt` files
- time values come from the first column and are scaled using the configured global `time_resolution`
- batch loading can apply timestamp offsets; if timestamp parsing is not usable, the backend falls back to continuous concatenation

**Not currently documented as supported:** richer file-format auto-detection, explicit-time-vector format detection, or arbitrary multi-column schema inference.

## Current user flow

### Load Data
The normal user path is the browser file picker. Recent sessions can be restored directly in the UI. Absolute-path loading exists on the backend API, but it should be treated as an internal/API-oriented path rather than the primary end-user workflow.

### Process & Detect
Users can optionally filter the signal, configure detection parameters, and run peak detection before moving into result analysis.

### Analyze Results
The analysis page is used for inspecting derived metrics and plots from detected peaks.

### Double Peak
The double-peak page is dedicated to paired-event analysis and related thresholds.

### Export
Export is handled on its own page rather than embedded as the main analysis workflow step.

## Architecture at a glance

- **`service/`** - FastAPI backend, file loading endpoints, export endpoints, task/progress services
- **`core/`** - numerical routines for loading, preprocessing, detection, correction, analysis, and export helpers
- **`ui-web/`** - Next.js / TypeScript user interface and workflow pages
- **`docs/`** - canonical product documentation, supplemental guides, historical notes, and engineering reports

## Running the application

### Windows launcher
From this checkout, start the application with:

```powershell
.\run_app.ps1
```

or double-click:

```bat
run_app.bat
```

The launcher reuses the local `.venv`, starts or reuses the backend on port `8765`, and opens the application at `http://127.0.0.1:8765/load/`.

### Run from source

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m service.app
```

### Frontend development

```powershell
cd ui-web
npm install
npm run dev
```

## Canonical documentation path

Start with these files:
- `docs/README.md` - authoritative documentation index
- `docs/USER_MANUAL.md` - end-user workflow and operating guidance
- `docs/user_interface.md` - route-by-route UI guide with screenshots
- `docs/mathematical_reference.md` - mathematical and scientific basis

Supplemental guides under `docs/guides/` remain available for scoped reference, but they are not the primary reader path.

## Repository boundaries for Phase 1

This repository phase is focused on documentation clarity and internal/commercial posture.

It does **not** currently include:
- a public/open-source support promise
- a general public issue/support workflow
- feature work for richer explicit-time-vector upload formats
- a finalized commercial license grant

## Future follow-up (not implemented in Phase 1)

A later planning phase may define how the tool should handle richer upload formats, including files that carry more explicit timing metadata. That work is not represented as current functionality.

## Internal contact

For internal licensing, review coordination, or product questions, contact the repository owner directly through your existing internal communication channel.
