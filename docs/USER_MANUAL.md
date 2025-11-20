# Peak Analysis Tool User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Workflow Overview](#workflow-overview)
4. [Loading Data](#loading-data)
5. [Processing & Detection](#process--detect)
6. [Analysis & Visualization](#analysis--export)
7. [Double Peak Analysis](#double-peak-analysis)
8. [Exporting Results](#exporting-results)
9. [Preferences](#preferences)

---

## Introduction

The Peak Analysis Tool is a high-precision application for detecting, analyzing, and characterizing peaks in time-series data. It is designed for scientific use cases, such as single-molecule fluorescence or particle counting, where accurate peak detection and noise filtering are critical.

## Getting Started

### Installation
- **Windows**: Run `PeakService.exe`.
- **Source**: Run `python -m service.app`.

The application interface will open automatically in your default web browser at `http://localhost:8765`.

## Workflow Overview

The application follows a structured workflow:
1. **Load**: Import raw data files.
2. **Process & Detect**: Filter noise and identify peaks.
3. **Analyze**: View time-series results and distributions.
4. **Export**: Save data and plots.

## Loading Data

Navigate to the **Load Data** page to begin.

- **File Selection**: Click "Select Files" to choose `.txt`, `.xls`, or `.xlsx` files.
- **Protocol Information**: Enter metadata about your experiment (Date, Setup, Sample ID, Laser Power, etc.) to keep it associated with the dataset.
- **Settings**:
  - **Time Resolution**: Set the time per sample (e.g., 0.1 ms).
  - **Photon Correction**: Enable dead-time correction if using a photon counter. Enter the dead time in nanoseconds.

Recent sessions are saved and can be reloaded with a single click from the "Recent Sessions" list.

## Process & Detect

This is the core analysis step.

### 1. Signal Filtering
Apply a digital filter to reduce noise before detection.
- **Filter Type**:
  - **Butterworth**: General-purpose noise reduction.
  - **Savitzky-Golay**: Preserves peak shape better while smoothing.
- **Parameters**:
  - **Cutoff Frequency (Hz)**: Frequencies above this are removed.
  - **Order**: Sharpness of the filter cutoff.
  - **Window Length**: Number of points for smoothing (Savitzky-Golay).

### 2. Peak Detection
Configure algorithms to find valid peaks.
- **Prominence**: How much a peak stands out from the surrounding baseline.
- **Min Distance**: Minimum samples between peaks.
- **Width Min/Max**: Accepted width range in milliseconds.
- **Rel Height**: The relative height (0.0-1.0) at which width is measured.
- **Prominence Ratio**: Filters noise by comparing prominence to absolute amplitude.

Click **Detect Peaks** to run the algorithm. Use **Auto Threshold** for an initial estimate.

## Analysis & Export

View the detected peaks in context of the full time series.

- **Interactive Plot**: Zoom and pan to inspect individual peaks.
- **Throughput**: See the rate of peaks over time.
- **Histograms**: Analyze distributions of peak amplitudes and intervals.
- **Export**: Use the integrated Export controls to save peak data (CSV) or download the current chart view as an image.

## Double Peak Analysis

For experiments involving paired events (e.g., double-labeled particles).

- **Pairing Logic**: Peaks are paired based on their temporal proximity.
- **Metrics**:
  - **Distance**: Time between paired peaks.
  - **Prominence Ratio**: Ratio of the second peak's intensity to the first.
  - **Width Ratio**: Comparison of peak shapes.
- **Thresholds**: Set min/max ranges for these metrics to filter valid pairs.

## Exporting Results

Save your work for publication or further analysis.

- **Data Formats**: CSV, Excel, Text.
- **Metadata**: Option to include protocol details in the header.
- **Images**: Click the **Camera** icon on any chart to save a high-resolution PNG.

## Preferences

Customize the application appearance (Light/Dark mode) and default analysis parameters.
