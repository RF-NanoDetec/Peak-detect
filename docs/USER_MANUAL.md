# Peak Analysis Tool User Manual

## Purpose

This manual describes how to operate the current browser-based Peak Analysis Tool as it exists today. It is the canonical operator guide for the current private/internal-use phase.

## Workflow overview

The application follows this workflow:
1. **Load Data**
2. **Process & Detect**
3. **Analyze Results**
4. **Double Peak**

**Export** is available as a separate page/action once the analysis results exist.

## Before you begin

### Supported input files
- `.txt`
- `.xls`
- `.xlsx`

### How input files are interpreted
- the backend reads the **first two columns only**
- for `.txt`, parsing is **tab-delimited**
- blank lines and `# comments` are skipped in `.txt`
- the first column is treated as time-like input and is scaled by the configured global `time_resolution`
- the second column is treated as amplitude / signal magnitude

### Important current limitation
The application does **not** currently advertise automatic schema detection for richer file layouts. If your data uses a different structure, it should be normalized into the currently supported two-column layout before use.

## 1. Load Data

**Primary route:** `/load`

This is the normal starting point for users.

### Recommended loading flow
1. Click **Select Files**
2. Choose one or more `.txt`, `.xls`, or `.xlsx` files
3. Review the local preview of the first selected file
4. Set the time resolution
5. Optionally enable photon-counter dead-time correction
6. Optionally fill in protocol metadata
7. Click **Load Files**

### Recent sessions
The UI can restore previously used file groups through the **Recent Sessions** list.

### Internal/API-oriented loading path
The backend also exposes an absolute-path file-loading endpoint. This is not the primary end-user workflow and should be treated as an internal/API-oriented path.

## 2. Process & Detect

**Primary route:** `/preprocess`

This step combines preprocessing setup and peak detection control.

### Filtering options
- **None** ? pass-through
- **Butterworth** ? low-pass filtering
- **Savitzky-Golay** ? smoothing with local polynomial fitting

### Detection controls
Users can configure:
- prominence threshold
- minimum distance
- width constraints
- relative height
- prominence ratio

Auto-threshold and related helper controls are available for faster initial setup.

## 3. Analyze Results

**Primary route:** `/analyze`

The analysis page is for inspecting the processed results after peak detection.

Typical uses:
- review signal plots and detected peaks
- inspect summary metrics
- inspect histogram-style or derived views tied to the detected results

## 4. Double Peak

**Primary route:** `/double`

Use this page to examine consecutive peak pairs and apply double-peak-specific thresholds.

Typical outputs include:
- distance between paired events
- pair prominence ratios
- pair width ratios
- filtered subsets of candidate pairs

## 5. Export

**Primary route:** `/export`

Export is intentionally handled on a separate page.

Current export scope includes:
- single-peak outputs
- double-peak outputs
- combined outputs
- metadata-aware table export paths driven by the dedicated export page

Chart image export exists in the chart views themselves and should not be confused with the dedicated `/export` page.

## Protocol metadata

During loading, the tool can store experiment metadata such as:
- measurement date / start time
- setup
- sample number
- particle
- concentration
- buffer and buffer concentration
- ND filter
- laser power
- stamp
- notes

These fields are optional but useful for internal traceability.

## Photon-counter correction

If your detector exhibits dead-time effects, enable dead-time correction during loading. The configured dead time is applied during the backend load phase and stored alongside correction metadata.

## Recommended operator checks

Before trusting a run, confirm:
- the file preview looks structurally correct
- the time resolution matches the experiment
- filtering has not erased real peaks
- detection thresholds are not dominated by noise
- exports correspond to the correct result set

## Current boundaries

This manual describes the **current** product behavior only. It does not promise:
- support for arbitrary file schemas
- automatic richer-format detection
- a public/open-source support workflow
