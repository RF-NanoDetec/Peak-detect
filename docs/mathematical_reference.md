# Mathematical Reference and Processing Notes

This document summarizes the scientific and algorithmic basis of the current Peak Analysis Tool implementation.

## 1. Signal ingestion assumptions

The current loader assumes a simple two-column input model:
- column 1: time-like values
- column 2: amplitude values

For text input, parsing is tab-delimited and ignores blank lines and comment lines beginning with `#`.

The first column is scaled into seconds using the configured global `time_resolution`:

$$
t_i = x_i \cdot \Delta t
$$

where:
- $x_i$ is the raw value read from column 1
- $\Delta t$ is the configured `time_resolution`

In batch mode, timestamp-aware offsets may be applied. If timestamp handling is not usable, segments are concatenated continuously using the previous segment boundary.

## 2. Peak detection

Peak detection is built on local-maximum identification with configurable prominence, spacing, and width constraints.

The implementation uses `scipy.signal.find_peaks` and associated peak-property utilities.

### 2.1 Prominence

Peak prominence measures how far a peak rises above its surrounding baseline:

$$
P_i = H_i - B_i
$$

where $H_i$ is the peak height and $B_i$ is the effective contour/baseline level returned by the peak-property calculation.

### 2.2 Prominence-ratio filtering

The tool also supports filtering on the ratio between prominence and absolute amplitude:

$$
R_{prom} = \frac{P_i}{A_i}
$$

where $A_i$ is the absolute amplitude of the peak. This helps suppress small sub-peaks riding on larger structures.

### 2.3 Peak width

Width is measured at a configurable relative height:

$$
L = H - r(H-B)
$$

where:
- $H$ is the peak height
- $B$ is the baseline
- $r$ is `rel_height`

Linear interpolation is used for sub-sample crossing estimates.

## 3. Preprocessing

### 3.1 Butterworth low-pass filter

For low-pass filtering, the tool uses a Butterworth response:

$$
|H(j\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2N}}
$$

where $\omega_c$ is the cutoff frequency and $N$ is the filter order.

### 3.2 Savitzky-Golay smoothing

Savitzky-Golay preprocessing performs a local polynomial fit over a sliding window:

$$
y_j = \sum_{i=-M}^{M} C_i x_{j+i}
$$

where the coefficients $C_i$ are derived from the least-squares polynomial fit.

## 4. Dead-time correction

When enabled, the application applies a non-paralyzable detector dead-time correction during loading.

Let:
- $R_{measured}$ be the measured count rate
- $\tau_D$ be the detector dead time
- $R_{true}$ be the corrected rate

Then:

$$
R_{true} = \frac{R_{measured}}{1 - R_{measured}\tau_D}
$$

The same factor is applied to the raw counts/signal. Near saturation, the correction factor is clamped to avoid numerical instability.

## 5. Decimation and visualization

Large datasets may be decimated for plotting responsiveness, but this is a visualization concern rather than a scientific change to the underlying result arrays. Analysis and export logic operate on the full-resolution arrays stored by the backend.

## 6. Double-peak analysis

Double-peak analysis operates on consecutive peaks and evaluates pairwise quantities such as:
- peak-to-peak distance
- start-to-start distance
- amplitude ratio
- width ratio

A pair is considered a matching double-peak candidate only when it falls inside the configured threshold ranges for the relevant metrics.

## 7. Current model boundary

This reference describes the **current implementation only**. It does not claim support for automatic schema detection or richer explicit-time-vector ingestion beyond the existing two-column loader assumptions.
