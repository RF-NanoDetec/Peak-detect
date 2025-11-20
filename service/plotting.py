"""
Server-side plotting module using matplotlib to match original GUI styling.
Generates static images for web display.

OPTIMIZED: Uses smart decimation for faster rendering while preserving visual quality.
"""

import time
import io
import base64
from typing import Any, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import logging
from core.data_utils import decimate_min_max

logger = logging.getLogger(__name__)


def generate_preprocessing_plot(time_data, amplitude_data, filtered_amplitude=None):
    """
    Generate preprocessing plot matching original GUI style (OPTIMIZED).
    
    OPTIMIZATION: Uses min-max decimation for faster rendering while preserving peaks.
    
    Args:
        time_data: Time array in seconds
        amplitude_data: Original amplitude array
        filtered_amplitude: Filtered amplitude array (optional)
    
    Returns:
        base64 encoded PNG image
    """
    try:
        # OPTIMIZATION: Decimate for faster plotting (preserves visual peaks)
        MAX_PLOT_POINTS = 20000
        original_len = len(time_data)
        
        if len(time_data) > MAX_PLOT_POINTS:
            logger.info(f"Decimating {original_len} points to {MAX_PLOT_POINTS} for plotting")
            t_plot_sec, amp_decimated = decimate_min_max(time_data, amplitude_data, MAX_PLOT_POINTS)
            t_plot = t_plot_sec / 60  # Convert to minutes
            
            if filtered_amplitude is not None:
                _, filtered_decimated = decimate_min_max(time_data, filtered_amplitude, MAX_PLOT_POINTS)
            else:
                filtered_decimated = None
        else:
            t_plot = time_data / 60
            amp_decimated = amplitude_data
            filtered_decimated = filtered_amplitude
        
        # Create figure
        fig = Figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Set background colors to match GUI
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#1e1e1e')
        
        # Plot raw data with thin line (matching original: linewidth=0.05)
        raw_line_color = '#888888'  # Gray for raw
        ax.plot(
            t_plot,
            amp_decimated,
            color=raw_line_color,
            linewidth=0.05,
            alpha=0.4,
            label=f'Original Data ({original_len:,} points)'
        )
        
        # Plot filtered data if provided
        if filtered_decimated is not None:
            filtered_line_color = '#5b9bd5'  # Blue for filtered
            ax.plot(
                t_plot,
                filtered_decimated,
                color=filtered_line_color,
                linewidth=0.05,
                alpha=0.9,
                label=f'Filtered Data ({original_len:,} points)'
            )
        
        # Styling
        ax.set_xlabel('Time (min)', color='white', fontsize=10)
        ax.set_ylabel('Counts', color='white', fontsize=10)
        ax.set_title('Signal Comparison' if filtered_decimated is not None else 'Signal Preview', 
                     color='white', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.tick_params(colors='white', labelsize=9)
        
        # Set tight limits
        ax.set_xlim(t_plot.min(), t_plot.max())
        
        # Legend with thicker lines for visibility
        legend = ax.legend(loc='upper right', fontsize=9)
        legend.get_frame().set_facecolor('#2b2b2b')
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Convert to base64 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                   bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        logger.info(f"Generated preprocessing plot: {original_len} points (decimated to {len(t_plot)} for rendering)")
        return img_base64
        
    except Exception as e:
        logger.error(f"Error generating preprocessing plot: {e}")
        raise


def generate_detection_plot_with_peaks(time_data, amplitude_data, peak_times, peak_amplitudes):
    """
    Generate detection plot with signal and peak markers overlay (OPTIMIZED).
    
    OPTIMIZATION: Uses min-max decimation for faster rendering while preserving peaks.
    Peak markers are plotted separately to ensure visibility.
    
    Args:
        time_data: Time array in seconds
        amplitude_data: Amplitude array
        peak_times: Array of peak time values (seconds)
        peak_amplitudes: Array of peak amplitude values
    
    Returns:
        base64 encoded PNG image
    """
    try:
        # OPTIMIZATION: Decimate signal for faster plotting
        MAX_PLOT_POINTS = 20000
        original_len = len(time_data)
        
        if len(time_data) > MAX_PLOT_POINTS:
            logger.info(f"Decimating {original_len} points to {MAX_PLOT_POINTS} for detection plot")
            t_plot_sec, amp_decimated = decimate_min_max(time_data, amplitude_data, MAX_PLOT_POINTS)
            t_plot = t_plot_sec / 60  # Convert to minutes
        else:
            t_plot = time_data / 60
            amp_decimated = amplitude_data
        
        # Create figure
        fig = Figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Set background colors to match dark theme
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#1e1e1e')
        
        # Plot signal line
        signal_color = '#5b9bd5'  # Blue for signal
        ax.plot(
            t_plot,
            amp_decimated,
            color=signal_color,
            linewidth=1.0,
            alpha=0.8,
            label=f'Signal ({original_len:,} points)'
        )
        
        # Plot peak markers (convert peak times to minutes)
        if len(peak_times) > 0:
            peak_times_min = np.array(peak_times) / 60
            ax.plot(
                peak_times_min,
                peak_amplitudes,
                marker='x',
                color='#ff6b6b',  # Red for peaks
                markersize=6,
                linestyle='None',
                markeredgewidth=1.5,
                label=f'Peaks ({len(peak_times):,})',
                zorder=10  # Ensure peaks are drawn on top
            )
        
        # Styling
        ax.set_xlabel('Time (min)', color='white', fontsize=10)
        ax.set_ylabel('Counts', color='white', fontsize=10)
        ax.set_title(f'Peak Detection ({len(peak_times):,} peaks)', 
                     color='white', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.tick_params(colors='white', labelsize=9)
        
        # Set tight limits
        ax.set_xlim(t_plot.min(), t_plot.max())
        
        # Legend
        legend = ax.legend(loc='upper right', fontsize=9)
        legend.get_frame().set_facecolor('#2b2b2b')
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Convert to base64 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                   bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        logger.info(f"Generated detection plot: {original_len} points, {len(peak_times)} peaks")
        return img_base64
        
    except Exception as e:
        logger.error(f"Error generating detection plot: {e}")
        raise


def generate_peak_regions_plot(time_data, amplitude_data, filtered_amplitude, peaks, peak_properties, offset=0):
    """
    Generate plot showing individual peaks in 2x5 grid (matching original GUI).
    
    Args:
        time_data: Time array in seconds
        amplitude_data: Original amplitude array
        filtered_amplitude: Filtered amplitude array
        peaks: Array of peak indices
        peak_properties: Dict containing peak properties (widths, left_ips, right_ips, width_heights)
        offset: Offset for cycling through peaks
    
    Returns:
        base64 encoded PNG image
    """
    try:
        if len(peaks) == 0:
            logger.warning("No peaks to plot")
            return None
        
        # Match original: show 10 peaks from different segments
        num_segments = 10
        total_peaks = len(peaks)
        segment_size = total_peaks // num_segments
        
        # Ensure offset is valid
        offset = offset % total_peaks
        
        # Select peaks from different segments (matching original logic)
        selected_peak_indices = []
        for i in range(num_segments):
            segment_idx = (i * segment_size + offset) % total_peaks
            if segment_idx < total_peaks:
                selected_peak_indices.append(segment_idx)
        
        # Create figure (matching original: 8x6 inches)
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')
        
        # Get window sizes based on peak widths (matching original)
        widths = peak_properties.get('widths', np.ones(len(peaks)))
        window = np.round(3 * widths, 0).astype(int)
        
        # Create grid layout: 2 rows, 5 cols = 10 peaks
        for plot_idx, peak_idx in enumerate(selected_peak_indices):
            if plot_idx >= 10:  # Safety limit
                break
                
            ax = fig.add_subplot(2, 5, plot_idx + 1)
            ax.set_facecolor('#1e1e1e')
            
            # Calculate window around peak (matching original)
            i = peak_idx
            start_idx = max(0, peaks[i] - window[i])
            end_idx = min(len(time_data), peaks[i] + window[i])
            
            # Extract data segment
            xData = time_data[start_idx:end_idx]
            yData_filtered = filtered_amplitude[start_idx:end_idx]
            yData_raw = amplitude_data[start_idx:end_idx]
            
            if len(xData) == 0:
                continue
            
            # Subtract background (matching original)
            background = np.min(yData_filtered)
            yData_filtered = yData_filtered - background
            yData_raw = yData_raw - background
            
            # Convert time to milliseconds relative to window start (matching original)
            xData_ms = (xData - xData[0]) * 1e3
            
            # Plot filtered signal (orange in original)
            ax.plot(xData_ms, yData_filtered,
                   color='#ff8c42',  # Orange
                   linewidth=0.7,
                   alpha=0.8,
                   label='Filtered')
            
            # Plot raw signal (blue in original, semi-transparent)
            ax.plot(xData_ms, yData_raw,
                   color='#5b9bd5',  # Blue
                   linewidth=0.5,
                   alpha=0.6,
                   label='Raw')
            
            # Mark peak (red × in original)
            peak_time_ms = (time_data[peaks[i]] - xData[0]) * 1e3
            peak_height = filtered_amplitude[peaks[i]] - background
            ax.plot(peak_time_ms, peak_height,
                   marker='x',
                   color='#ff6b6b',  # Red
                   markersize=8,
                   linestyle='None')
            
            # Add width indicators (green horizontal lines in original)
            left_idx = int(peak_properties["left_ips"][i])
            right_idx = int(peak_properties["right_ips"][i])
            width_height = peak_properties["width_heights"][i] - background
            
            ax.hlines(y=width_height,
                     xmin=(time_data[left_idx] - xData[0]) * 1e3,
                     xmax=(time_data[right_idx] - xData[0]) * 1e3,
                     color='#4caf50',  # Green
                     linestyles='-',
                     linewidth=1.0)
            
            # Styling (matching original)
            ax.set_xlabel('Time (ms)', color='white', fontsize=8)
            ax.set_ylabel('Amplitude (counts)', color='white', fontsize=8)
            ax.set_title(f'Peak {peak_idx + 1}', color='white', fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.3, color='gray')
            ax.tick_params(colors='white', labelsize=7)
        
        fig.tight_layout()
        
        # Convert to base64 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                   bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        logger.info(f"Generated peak inspection plot: showing peaks {offset}-{offset+10} of {total_peaks}")
        return img_base64
        
    except Exception as e:
        logger.error(f"Error generating peak regions plot: {e}")
        raise


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
    # Clamp to reasonable range
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
    
    Args:
        data: Input data array
        bin_count: Number of bins (optional, auto-calculated if not provided)
        range_override: Optional range to filter data
        log_scale: Use logarithmic binning
    
    Returns:
        Dictionary with bins, counts, and bin_edges
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
    logger.debug(f"Calculating histogram: {len(data_array)} data points, {bins_requested} bins, "
                 f"log_scale={log_scale}")
    
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
    
    # Ensure all values are finite before sending to frontend
    bin_centers = np.nan_to_num(bin_centers, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    bins = np.nan_to_num(bins, nan=0.0, posinf=0.0, neginf=0.0)
    
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        logger.info(f"Histogram calculation took {elapsed:.4f}s for {len(data_array)} points")
    
    return {
        "bins": bin_centers.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bins.tolist()
    }


def generate_peak_histogram_data(peak_amplitudes, peak_widths_ms, peak_intervals_ms, config=None):
    """
    Calculate histogram data for peak statistics.
    
    Args:
        peak_amplitudes: Array of peak amplitudes (logarithmic scale)
        peak_widths_ms: Array of peak widths in milliseconds (linear scale)
        peak_intervals_ms: Array of intervals between peaks in milliseconds (logarithmic scale)
    
        config: Optional configuration for binning behaviour.
    
    Returns:
        dict with histogram data for each distribution
    """
    config = config or {}
    metrics_config: Dict[str, Any] = config.get("metrics", {})
    range_overrides = config.get("range_overrides", {}) or {}
    bin_count = config.get("bin_count")
    
    def metric_hist(values, metric_key: str):
        try:
            numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
            if len(numeric_values) == 0:
                logger.debug(f"No valid numeric values for {metric_key} histogram")
                return {"bins": [], "counts": []}
            
            metric_cfg = metrics_config.get(metric_key, {})
            x_scale = metric_cfg.get("xScale") or metric_cfg.get("x_scale") or "linear"
            range_override = range_overrides.get(metric_key) or metric_cfg.get("range") or metric_cfg.get("range_override")
            
            logger.debug(f"Calculating {metric_key} histogram: {len(numeric_values)} values, scale={x_scale}, bins={bin_count}")
            
            return calculate_histogram_bins(
                numeric_values,
                bin_count=bin_count,
                range_override=range_override,
                log_scale=x_scale == "log",
            )
        except Exception as e:
            logger.error(f"Error calculating {metric_key} histogram: {e}", exc_info=True)
            return {"bins": [], "counts": []}
    
    try:
        if len(peak_amplitudes) == 0:
            logger.warning("No peak data to calculate histograms")
            return None

        amplitude_hist = metric_hist(peak_amplitudes, "amplitude")
        width_hist = metric_hist(peak_widths_ms, "width")
        interval_hist = metric_hist(peak_intervals_ms, "interval")

        logger.info(f"Generated histograms: amplitude={len(amplitude_hist['bins'])} bins, "
                   f"width={len(width_hist['bins'])} bins, interval={len(interval_hist['bins'])} bins")

        return {
            "amplitude": amplitude_hist,
            "width": width_hist,
            "interval": interval_hist
        }
        
    except Exception as e:
        logger.error(f"Error calculating histogram data: {e}", exc_info=True)
        raise


def generate_peak_histograms(peak_amplitudes, peak_widths_ms, peak_intervals_ms):
    """
    Generate three histograms side by side for peak statistics.
    
    DEPRECATED: Use generate_peak_histogram_data instead for uPlot integration.
    
    Args:
        peak_amplitudes: Array of peak amplitudes (logarithmic scale)
        peak_widths_ms: Array of peak widths in milliseconds (linear scale)
        peak_intervals_ms: Array of intervals between peaks in milliseconds (logarithmic scale)
    
    Returns:
        base64 encoded PNG image
    """
    try:
        if len(peak_amplitudes) == 0:
            logger.warning("No peak data to plot histograms")
            return None
        
        # Create figure with three subplots side by side
        fig = Figure(figsize=(18, 5), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')
        
        # Calculate reasonable binning using Freedman-Diaconis rule or Scott's rule
        def calculate_bins(data, is_log=False):
            if len(data) == 0:
                return 20
            # Convert to numpy array to ensure .max() and .min() work
            data = np.asarray(data)
            if is_log:
                # For log scale, use log-transformed data
                data_positive = data[data > 0]
                if len(data_positive) == 0:
                    return 20
                log_data = np.log10(data_positive)
                iqr = np.percentile(log_data, 75) - np.percentile(log_data, 25)
                if iqr == 0:
                    return 20
                bin_width = 2 * iqr / (len(data_positive) ** (1/3))
                if bin_width == 0:
                    return 20
                num_bins = int((log_data.max() - log_data.min()) / bin_width)
                return max(10, min(50, num_bins))
            else:
                # For linear scale, use standard Freedman-Diaconis
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                if iqr == 0:
                    return 20
                bin_width = 2 * iqr / (len(data) ** (1/3))
                if bin_width == 0:
                    return 20
                num_bins = int((data.max() - data.min()) / bin_width)
                return max(10, min(50, num_bins))
        
        # Histogram 1: Amplitude (logarithmic)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_facecolor('#1e1e1e')
        
        amplitudes_positive = [a for a in peak_amplitudes if a > 0]
        if len(amplitudes_positive) > 0:
            # Use log scale directly with matplotlib
            bins_amp = calculate_bins(amplitudes_positive, is_log=True)
            # Create log-spaced bins
            log_min = np.log10(min(amplitudes_positive))
            log_max = np.log10(max(amplitudes_positive))
            log_bins = np.logspace(log_min, log_max, bins_amp)
            ax1.hist(
                amplitudes_positive,
                bins=log_bins,
                color='#5b9bd5',
                edgecolor='#3a7ba5',
                alpha=0.8,
                linewidth=0.5
            )
            ax1.set_xscale('log')
        else:
            ax1.text(0.5, 0.5, 'No positive amplitudes', 
                    ha='center', va='center', color='white', transform=ax1.transAxes)
        
        ax1.set_xlabel('Amplitude (log scale)', color='white', fontsize=10)
        ax1.set_ylabel('Frequency', color='white', fontsize=10)
        ax1.set_title('Amplitude Distribution', color='white', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax1.tick_params(colors='white', labelsize=9)
        
        # Histogram 2: Width (linear)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_facecolor('#1e1e1e')
        
        if len(peak_widths_ms) > 0:
            widths_positive = [w for w in peak_widths_ms if w > 0]
            if len(widths_positive) > 0:
                bins_width = calculate_bins(widths_positive, is_log=False)
                ax2.hist(
                    widths_positive,
                    bins=bins_width,
                    color='#ff8c42',
                    edgecolor='#cc6f33',
                    alpha=0.8,
                    linewidth=0.5
                )
            else:
                ax2.text(0.5, 0.5, 'No valid widths', 
                        ha='center', va='center', color='white', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'No width data', 
                    ha='center', va='center', color='white', transform=ax2.transAxes)
        
        ax2.set_xlabel('Width (ms)', color='white', fontsize=10)
        ax2.set_ylabel('Frequency', color='white', fontsize=10)
        ax2.set_title('Peak Width Distribution', color='white', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax2.tick_params(colors='white', labelsize=9)
        
        # Histogram 3: Distance between peaks (logarithmic)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_facecolor('#1e1e1e')
        
        if len(peak_intervals_ms) > 0:
            intervals_positive = [i for i in peak_intervals_ms if i > 0]
            if len(intervals_positive) > 0:
                # Use log scale directly with matplotlib
                bins_intervals = calculate_bins(intervals_positive, is_log=True)
                # Create log-spaced bins
                log_min = np.log10(min(intervals_positive))
                log_max = np.log10(max(intervals_positive))
                log_bins = np.logspace(log_min, log_max, bins_intervals)
                ax3.hist(
                    intervals_positive,
                    bins=log_bins,
                    color='#4caf50',
                    edgecolor='#3d8b40',
                    alpha=0.8,
                    linewidth=0.5
                )
                ax3.set_xscale('log')
            else:
                ax3.text(0.5, 0.5, 'No valid intervals', 
                        ha='center', va='center', color='white', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'No interval data', 
                    ha='center', va='center', color='white', transform=ax3.transAxes)
        
        ax3.set_xlabel('Distance Between Peaks (ms, log scale)', color='white', fontsize=10)
        ax3.set_ylabel('Frequency', color='white', fontsize=10)
        ax3.set_title('Peak Distance Distribution', color='white', fontsize=11)
        ax3.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax3.tick_params(colors='white', labelsize=9)
        
        fig.tight_layout()
        
        # Convert to base64 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                   bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        logger.info(f"Generated peak histograms: {len(peak_amplitudes)} peaks")
        return img_base64
        
    except Exception as e:
        logger.error(f"Error generating peak histograms: {e}")
        raise
