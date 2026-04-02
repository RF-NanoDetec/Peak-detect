# User Interface Guide

This document maps the current web UI to the supported analysis workflow. It is the canonical screenshot-oriented guide for the current application.

## Workflow map

The left sidebar currently exposes these workflow steps:
1. **Load Data** (`/load`)
2. **Process & Detect** (`/preprocess`)
3. **Analyze Results** (`/analyze`)
4. **Double Peak** (`/double`)

Export is handled separately on the dedicated `/export` page.

## 1. Load Data

**Route:** `/load`

![Load Data Interface](figures/snapshots/ui_load_data.png)

### What users do here
- select local files with the browser file picker
- restore a recent file group from **Recent Sessions**
- review a local preview of the first selected file
- configure time resolution
- enable or disable dead-time correction
- capture protocol metadata

### What this page is not
Although the backend supports absolute-path loading, the page is designed around browser-side file selection rather than manual path entry as the primary user experience.

## 2. Process & Detect

**Route:** `/preprocess`

![Process & Detect Interface](figures/snapshots/ui_preprocess.png)

### What users do here
- choose a preprocessing method
- tune filter parameters
- configure peak-detection settings
- run preprocessing and peak detection

### Main controls
- filter type (`None`, `Butterworth`, `Savitzky-Golay`)
- cutoff frequency / smoothing parameters
- prominence threshold
- minimum distance
- width bounds
- prominence ratio

## 3. Analyze Results

**Route:** `/analyze`

![Analyze Interface](figures/snapshots/ui_analyze.png)

### What users do here
- inspect detected-peak results in context
- review analysis metrics
- review derived visual summaries tied to the current detection result

### Important boundary
This page is for review and interpretation. The dedicated export workflow lives on `/export`.

## 4. Double Peak

**Route:** `/double`

![Double Peak Interface](figures/snapshots/ui_double_peak.png)

### What users do here
- inspect consecutive peak pairs
- apply pair-specific thresholds
- review scatter/histogram-style double-peak views

## 5. Export

**Route:** `/export`

Export is a separate route and should be documented that way.

Current export behavior supports:
- export of single-peak datasets
- export of double-peak datasets
- combined export modes
- metadata-aware export paths

## Screenshot notes

The screenshots above reflect the current web workflow and should remain aligned with the route names used in the sidebar.
