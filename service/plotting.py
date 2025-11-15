"""
Server-side plotting module using matplotlib to match original GUI styling.
Generates static images for web display.

OPTIMIZED: Uses smart decimation for faster rendering while preserving visual quality.
"""

import io
import base64
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


