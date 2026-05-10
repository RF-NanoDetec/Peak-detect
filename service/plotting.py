"""
Histogram helpers for chart data returned to the web UI.

Interactive traces are rendered client-side with uPlot. Static Matplotlib image
generation was removed with the obsolete peak-inspection and plot-image export
paths.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_range(values):
    if values is None:
        return None
    if isinstance(values, dict):
        return (values.get("min"), values.get("max"))
    if isinstance(values, (list, tuple)) and len(values) == 2:
        return (values[0], values[1])
    return None


def _estimate_bin_count(data) -> int:
    """
    Estimate optimal bin count using Freedman-Diaconis rule.
    Returns a value between 5 and 500 to prevent extreme binning.
    """
    n = len(data)
    if n <= 1:
        return 10

    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr == 0 or not np.isfinite(iqr):
        return max(10, min(100, int(np.cbrt(n))))

    bin_width = 2 * iqr / (n ** (1 / 3))
    if bin_width <= 0 or not np.isfinite(bin_width):
        return 20

    data_range = data.max() - data.min()
    if data_range <= 0 or not np.isfinite(data_range):
        return 20

    bin_count = int(np.ceil(data_range / bin_width))
    return max(5, min(500, bin_count))


def calculate_histogram_bins(
    data,
    *,
    bin_count: int | None = None,
    range_override=None,
    log_scale: bool = False,
) -> Dict[str, Any]:
    """
    Calculate histogram bins and counts for given data.
    """
    if len(data) == 0:
        logger.debug("Empty data array for histogram")
        return {"bins": [], "counts": []}

    data_array = np.asarray(data, dtype=float)
    data_array = data_array[np.isfinite(data_array)]
    if data_array.size == 0:
        return {"bins": [], "counts": []}

    normalized_range = _normalize_range(range_override)
    if normalized_range:
        min_val, max_val = normalized_range
        if min_val is not None:
            data_array = data_array[data_array >= float(min_val)]
        if max_val is not None:
            data_array = data_array[data_array <= float(max_val)]
        if data_array.size == 0:
            return {"bins": [], "counts": []}

    effective_log = log_scale
    if effective_log:
        positive_data = data_array[data_array > 0]
        if positive_data.size == 0:
            return {"bins": [], "counts": []}
        data_array = positive_data

    min_value = data_array.min()
    max_value = data_array.max()
    if min_value == max_value:
        max_value = min_value + 1e-9

    bins_requested = int(bin_count) if bin_count else _estimate_bin_count(data_array)
    bins_requested = max(5, min(500, bins_requested))

    start_time = time.time()
    logger.debug(
        f"Calculating histogram: {len(data_array)} data points, {bins_requested} bins, "
        f"log_scale={log_scale}"
    )

    if effective_log:
        log_min = np.log10(min_value)
        log_max = np.log10(max_value)
        bins = np.logspace(log_min, log_max, bins_requested + 1, base=10)
    else:
        bins = np.linspace(min_value, max_value, bins_requested + 1)

    counts, _ = np.histogram(data_array, bins=bins)

    if effective_log:
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
    else:
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

    bin_centers = np.nan_to_num(bin_centers, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    bins = np.nan_to_num(bins, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = time.time() - start_time
    if elapsed > 0.1:
        logger.info(
            f"Histogram calculation took {elapsed:.4f}s for {len(data_array)} points"
        )

    return {
        "bins": bin_centers.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bins.tolist(),
    }


def generate_peak_histogram_data(
    peak_amplitudes, peak_widths_ms, peak_intervals_ms, config=None
):
    """
    Calculate histogram data for peak statistics.
    """
    config = config or {}
    metrics_config: Dict[str, Any] = config.get("metrics", {})
    range_overrides = config.get("range_overrides", {}) or {}
    bin_count = config.get("bin_count")

    def metric_hist(values, metric_key: str):
        try:
            numeric_values = [
                v for v in values if isinstance(v, (int, float)) and not np.isnan(v)
            ]
            if len(numeric_values) == 0:
                logger.debug(f"No valid numeric values for {metric_key} histogram")
                return {"bins": [], "counts": []}

            metric_cfg = metrics_config.get(metric_key, {})
            x_scale = metric_cfg.get("xScale") or metric_cfg.get("x_scale") or "linear"
            range_override = (
                range_overrides.get(metric_key)
                or metric_cfg.get("range")
                or metric_cfg.get("range_override")
            )

            logger.debug(
                f"Calculating {metric_key} histogram: {len(numeric_values)} values, scale={x_scale}, bins={bin_count}"
            )

            return calculate_histogram_bins(
                numeric_values,
                bin_count=bin_count,
                range_override=range_override,
                log_scale=x_scale == "log",
            )
        except Exception as e:
            logger.error(
                f"Error calculating {metric_key} histogram: {e}", exc_info=True
            )
            return {"bins": [], "counts": []}

    try:
        if len(peak_amplitudes) == 0:
            logger.warning("No peak data to calculate histograms")
            return None

        amplitude_hist = metric_hist(peak_amplitudes, "amplitude")
        width_hist = metric_hist(peak_widths_ms, "width")
        interval_hist = metric_hist(peak_intervals_ms, "interval")

        logger.info(
            f"Generated histograms: amplitude={len(amplitude_hist['bins'])} bins, "
            f"width={len(width_hist['bins'])} bins, interval={len(interval_hist['bins'])} bins"
        )

        return {
            "amplitude": amplitude_hist,
            "width": width_hist,
            "interval": interval_hist,
        }

    except Exception as e:
        logger.error(f"Error calculating histogram data: {e}", exc_info=True)
        raise
