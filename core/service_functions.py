"""
Service-facing, UI-free functions that wrap core logic for use by the FastAPI backend.

These functions avoid any tkinter/UI dependencies and return plain Python/numpy/pandas
objects that can be serialized by the API layer.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Reuse performant file loader from core without bringing in UI behavior
from .file_handler import load_single_file
from .timing import get_tracker


def load_data_from_paths(
    paths: List[str],
    mode: str = "single",
    timestamps: Optional[List[str]] = None,
    time_resolution: float = 1e-4,
    apply_dead_time_correction: bool = False,
    dead_time_ns: float = 43.0,
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load one or more data files and return combined arrays and a preview DataFrame.

    Parameters
    ----------
    paths : list of str
        File paths to load. Supports .txt, .xls, .xlsx.
    mode : str
        "single" or "batch". In batch mode and when timestamps are provided,
        time offsets are computed from the first timestamp.
    timestamps : list of str, optional
        Timestamps (strings) corresponding to each file when in batch mode.
        If provided, used to offset time of subsequent segments.
    time_resolution : float
        Time resolution in seconds per sample (e.g., 1e-4).
    apply_dead_time_correction : bool
        Whether to apply photon counter dead-time correction.
    dead_time_ns : float
        Detector dead time in nanoseconds (default: 43 ns).
    protocol : dict, optional
        Experiment protocol metadata.

    Returns
    -------
    dict
        {
            "time": np.ndarray (float32, seconds),
            "amplitude": np.ndarray (float32),
            "data": pandas.DataFrame with columns ['Time - Plot 0','Amplitude - Plot 0'],
            "loaded_files": List[str] (basenames),
            "count": int,
            "time_range": (float, float),
            "preview_head": str,  # stringified head for quick preview
            "protocol": dict or None,  # protocol metadata
            "corrections": dict or None  # correction info if applied
        }
    """
    if not paths:
        raise ValueError("No paths provided")

    # Get timing tracker if available
    tracker = get_tracker()

    # Load each file (parallelize across files to hide I/O latency)
    results: List[Dict[str, Any]] = []

    def load_single_with_timing(idx: int, path: str) -> Dict[str, Any]:
        phase_name = f"load_file_{idx + 1}"
        if tracker:
            tracker.start_phase(phase_name)
        try:
            return load_single_file(
                path,
                timestamps=timestamps,
                index=idx,
                time_resolution=time_resolution,
            )
        finally:
            if tracker:
                tracker.end_phase(phase_name)

    file_count = len(paths)
    if file_count == 1:
        results.append(load_single_with_timing(0, paths[0]))
    else:
        max_workers = min(file_count, max(2, (os.cpu_count() or 4)))
        results_buffer: List[Optional[Dict[str, Any]]] = [None] * file_count  # type: ignore
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(load_single_with_timing, idx, path): idx
                for idx, path in enumerate(paths)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                results_buffer[idx] = result
        results = [r for r in results_buffer if r is not None]

    # Sort by original index to preserve order
    if tracker:
        tracker.start_phase("sort_results")
    results.sort(key=lambda x: x["index"])
    if tracker:
        tracker.end_phase("sort_results")

    # Pre-allocate and concatenate
    if tracker:
        tracker.start_phase("preallocate_arrays")
    total_points = sum(len(r["time"]) for r in results)
    combined_times = np.zeros(total_points, dtype=np.float32)
    combined_amplitudes = np.zeros(total_points, dtype=np.float32)
    if tracker:
        tracker.end_phase("preallocate_arrays")

    # Compute offsets if needed
    if tracker:
        tracker.start_phase("concatenate_data")
    start_idx = 0
    for i, r in enumerate(results):
        time_data = r["time"]  # already seconds
        amp_data = r["amplitude"]
        n = len(time_data)

        if mode == "batch" and timestamps and i > 0:
            # If timestamps provided, offset by absolute time from first timestamp
            # Keep simple: treat provided times as seconds already if they parse as float
            # Otherwise, fall back to concatenation with continuous time.
            try:
                (
                    float(timestamps[0])
                    if isinstance(timestamps[0], (int, float, str))
                    else 0.0
                )
                current_ts = (
                    float(timestamps[i])
                    if isinstance(timestamps[i], (int, float, str))
                    else 0.0
                )
                time_offset = current_ts
                combined_times[start_idx : start_idx + n] = time_data + time_offset
            except Exception:
                # Fallback: continuous concatenation
                if i > 0:
                    segment_spacing = (
                        (time_data[1] - time_data[0])
                        if n > 1
                        else r.get("time_resolution", time_resolution)
                    )
                    time_offset = combined_times[start_idx - 1] + segment_spacing
                    combined_times[start_idx : start_idx + n] = time_data + time_offset
                else:
                    combined_times[start_idx : start_idx + n] = time_data
        else:
            if i > 0:
                segment_spacing = (
                    (time_data[1] - time_data[0])
                    if n > 1
                    else r.get("time_resolution", time_resolution)
                )
                time_offset = combined_times[start_idx - 1] + segment_spacing
                combined_times[start_idx : start_idx + n] = time_data + time_offset
            else:
                combined_times[start_idx : start_idx + n] = time_data

        combined_amplitudes[start_idx : start_idx + n] = amp_data
        start_idx += n
    if tracker:
        tracker.end_phase("concatenate_data")

    # Apply dead-time correction if requested
    correction_info = None
    if apply_dead_time_correction:
        if tracker:
            tracker.start_phase("dead_time_correction")
        from .photon_correction import apply_dead_time_correction as apply_correction

        combined_amplitudes, correction_info = apply_correction(
            combined_amplitudes, time_resolution, dead_time_ns
        )
        if tracker:
            tracker.end_phase("dead_time_correction")

    if tracker:
        tracker.start_phase("create_dataframe")
    df = pd.DataFrame(
        {
            "Time - Plot 0": combined_times,
            "Amplitude - Plot 0": combined_amplitudes,
        }
    )
    loaded_files = [os.path.basename(p) for p in paths]

    preview_head = df.head().to_string(index=False)
    time_min = float(df["Time - Plot 0"].min()) if len(df) else 0.0
    time_max = float(df["Time - Plot 0"].max()) if len(df) else 0.0
    if tracker:
        tracker.end_phase("create_dataframe")

    return {
        "time": combined_times,
        "amplitude": combined_amplitudes,
        "data": df,
        "loaded_files": loaded_files,
        "count": len(paths),
        "time_range": (time_min, time_max),
        "preview_head": preview_head,
        "protocol": protocol,
        "corrections": correction_info,
    }


def analyze_double_peaks_pure(
    peaks: np.ndarray,
    properties: Dict[str, np.ndarray],
    time_resolution: float,
    min_distance: float = 0.001,
    max_distance: float = 0.100,
    min_amp_ratio: float = 0.1,
    max_amp_ratio: float = 10.0,
    min_width_ratio: float = 0.1,
    max_width_ratio: float = 10.0,
) -> Dict[str, Any]:
    """
    Analyze consecutive peaks to identify double peak patterns.

    Parameters
    ----------
    peaks : np.ndarray
        Array of peak indices
    properties : dict
        Dictionary of peak properties (prominences, widths, left_ips, right_ips)
    time_resolution : float
        Time resolution in seconds per sample
    min_distance : float
        Minimum time between peaks (seconds)
    max_distance : float
        Maximum time between peaks (seconds)
    min_amp_ratio : float
        Minimum ratio of secondary to primary peak amplitude
    max_amp_ratio : float
        Maximum ratio of secondary to primary peak amplitude
    min_width_ratio : float
        Minimum ratio of secondary to primary peak width
    max_width_ratio : float
        Maximum ratio of secondary to primary peak width

    Returns
    -------
    dict
        Dictionary containing:
        - peak_pairs: list of dicts with info about each pair
        - total_pairs: int (total consecutive peak pairs analyzed)
        - double_peak_count: int (pairs meeting criteria)
    """
    if len(peaks) < 2:
        return {"peak_pairs": [], "total_pairs": 0, "double_peak_count": 0}

    peak_pairs = []
    double_peak_count = 0

    # Analyze each consecutive pair
    for i in range(len(peaks) - 1):
        current_peak = peaks[i]
        next_peak = peaks[i + 1]

        # Calculate peak-to-peak distance in seconds
        peak_distance = (next_peak - current_peak) * time_resolution

        # Calculate start-to-start distance
        current_start = properties["left_ips"][i]
        next_start = properties["left_ips"][i + 1]
        start_distance = (next_start - current_start) * time_resolution

        # Get peak properties
        current_amp = properties["prominences"][i]
        next_amp = properties["prominences"][i + 1]
        current_width = properties["widths"][i]
        next_width = properties["widths"][i + 1]

        # Calculate ratios
        amp_ratio = next_amp / current_amp if current_amp > 0 else 0
        width_ratio = next_width / current_width if current_width > 0 else 0

        # Check if meets criteria
        meets_criteria = (
            min_distance <= peak_distance <= max_distance
            and min_amp_ratio <= amp_ratio <= max_amp_ratio
            and min_width_ratio <= width_ratio <= max_width_ratio
        )

        if meets_criteria:
            double_peak_count += 1

        # Store pair information
        pair_info = {
            "primary_peak_idx": int(current_peak),
            "secondary_peak_idx": int(next_peak),
            "peak_distance_ms": float(peak_distance * 1000),
            "start_distance_ms": float(start_distance * 1000),
            "primary_amplitude": float(current_amp),
            "secondary_amplitude": float(next_amp),
            "amplitude_ratio": float(amp_ratio),
            "primary_width_ms": float(current_width * time_resolution * 1000),
            "secondary_width_ms": float(next_width * time_resolution * 1000),
            "width_ratio": float(width_ratio),
            "is_double_peak": bool(meets_criteria),
        }
        peak_pairs.append(pair_info)

    return {
        "peak_pairs": peak_pairs,
        "total_pairs": len(peaks) - 1,
        "double_peak_count": double_peak_count,
    }


def export_peaks_to_csv_data(
    time_values: np.ndarray,
    peaks: np.ndarray,
    properties: Dict[str, np.ndarray],
    time_resolution: float,
) -> pd.DataFrame:
    """
    Create a DataFrame with peak information for CSV export.

    Parameters
    ----------
    time_values : np.ndarray
        Array of time values in seconds
    peaks : np.ndarray
        Array of peak indices
    properties : dict
        Dictionary of peak properties (prominences, widths, etc.)
    time_resolution : float
        Time resolution in seconds per sample

    Returns
    -------
    pd.DataFrame
        DataFrame with peak information
    """
    if len(peaks) == 0:
        return pd.DataFrame()

    # Get peak times
    peak_times = time_values[peaks]

    # Get peak heights and widths
    peak_heights = properties.get("prominences", np.zeros(len(peaks)))
    peak_widths = properties.get("widths", np.zeros(len(peaks)))

    # Convert widths to milliseconds
    peak_widths_ms = peak_widths * time_resolution * 1000

    # Create DataFrame
    data = {
        "Time (s)": peak_times,
        "Amplitude": peak_heights,
        "Width (ms)": peak_widths_ms,
        "Width (samples)": peak_widths,
    }

    # Calculate intervals
    if len(peak_times) >= 2:
        intervals = np.zeros(len(peak_times))
        intervals[1:] = peak_times[1:] - peak_times[:-1]
        data["Interval (s)"] = intervals

    return pd.DataFrame(data)


def export_double_peaks_to_csv_data(
    double_peak_analysis: Dict[str, Any],
    time_values: np.ndarray,
    pair_indices: Optional[List[int]] = None,
    include_is_double: bool = True,
) -> pd.DataFrame:
    """
    Create a DataFrame with double peak information for CSV export.

    Parameters
    ----------
    double_peak_analysis : dict
        Result from analyze_double_peaks_pure
    time_values : np.ndarray
        Array of time values in seconds

    Returns
    -------
    pd.DataFrame
        DataFrame with double peak pair information
    """
    if not double_peak_analysis["peak_pairs"]:
        return pd.DataFrame()

    # Extract data from peak pairs
    pairs = double_peak_analysis["peak_pairs"]

    # Restrict to specific pairs if provided
    if pair_indices is not None and len(pair_indices) > 0:
        index_set = set(pair_indices)
        pairs = [p for i, p in enumerate(pairs) if i in index_set]

    data = {
        "Primary Peak Time (s)": [time_values[p["primary_peak_idx"]] for p in pairs],
        "Secondary Peak Time (s)": [
            time_values[p["secondary_peak_idx"]] for p in pairs
        ],
        "Peak Distance (ms)": [p["peak_distance_ms"] for p in pairs],
        "Start-to-Start Distance (ms)": [p["start_distance_ms"] for p in pairs],
        "Primary Peak Width (ms)": [p["primary_width_ms"] for p in pairs],
        "Secondary Peak Width (ms)": [p["secondary_width_ms"] for p in pairs],
        "Width Ratio": [p["width_ratio"] for p in pairs],
        "Amplitude Ratio": [p["amplitude_ratio"] for p in pairs],
    }

    if include_is_double:
        data["Is Double Peak"] = [p.get("is_double_peak", False) for p in pairs]

    return pd.DataFrame(data)


def export_unified_peaks_data(
    time_values: np.ndarray,
    peaks: np.ndarray,
    properties: Dict[str, np.ndarray],
    time_resolution: float,
    double_peak_analysis: Optional[Dict[str, Any]] = None,
    filter_double_peaks: bool = False,
    pair_indices: Optional[List[int]] = None,
    include_double_flags: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Create dataframes for single-peak data and, optionally, double-peak pair data.

    Returns a dictionary with:
        {
            "peaks_df": DataFrame of single peaks (with optional Is Double Peak mask),
            "double_df": DataFrame of double-peak pairs (may be empty)
        }
    """
    peaks_df = export_peaks_to_csv_data(time_values, peaks, properties, time_resolution)
    double_df = pd.DataFrame()

    if peaks_df.empty:
        return {"peaks_df": peaks_df, "double_df": double_df}

    # Add double peak mask and pair table if analysis is provided
    if double_peak_analysis and "peak_pairs" in double_peak_analysis:
        peak_pairs = double_peak_analysis["peak_pairs"]

        # Restrict to selected pair indices when provided
        if pair_indices is not None and len(pair_indices) > 0:
            pair_set = set(pair_indices)
            peak_pairs = [p for i, p in enumerate(peak_pairs) if i in pair_set]

        # Optionally build mask for peaks participating in selected pairs
        if include_double_flags:
            is_double_peak = np.zeros(len(peaks), dtype=bool)

            peak_idx_to_row = {peak_idx: i for i, peak_idx in enumerate(peaks)}
            for pair in peak_pairs:
                mark_peak = (
                    pair.get("is_double_peak", False)
                    if not filter_double_peaks
                    else True
                )
                if mark_peak:
                    primary_idx = pair["primary_peak_idx"]
                    secondary_idx = pair["secondary_peak_idx"]

                    if primary_idx in peak_idx_to_row:
                        is_double_peak[peak_idx_to_row[primary_idx]] = True
                    if secondary_idx in peak_idx_to_row:
                        is_double_peak[peak_idx_to_row[secondary_idx]] = True

            peaks_df["Is Double Peak"] = is_double_peak

        double_df = export_double_peaks_to_csv_data(
            {"peak_pairs": peak_pairs},
            time_values,
            pair_indices=None,  # already filtered above
            include_is_double=include_double_flags,
        )

        if filter_double_peaks:
            peaks_df = peaks_df[peaks_df.get("Is Double Peak", False)].copy()

    elif filter_double_peaks:
        # If filtering requested but no analysis provided, return empty peaks_df
        peaks_df = pd.DataFrame(columns=peaks_df.columns)

    return {"peaks_df": peaks_df, "double_df": double_df}
