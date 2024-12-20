import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks


def setup_seaborn():
    sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.2)
    sns.set_context("notebook", rc={"lines.linewidth": 1.0})

def decimate_for_plot( x, y, max_points=10000):
    """
    Intelligently reduce number of points for plotting while preserving important features
    
    Args:
        x: time array
        y: signal array
        max_points: maximum number of points to plot
    
    Returns:
        x_decimated, y_decimated: decimated arrays for plotting
    """
    if len(x) <= max_points:
        return x, y
    
    # Calculate decimation factor
    stride = len(x) // max_points
    
    # Initialize masks for important points
    mask = np.zeros(len(x), dtype=bool)
    
    # Include regularly spaced points
    mask[::stride] = True
    
    # Find peaks and include points around them
    peaks, _ = find_peaks(y, height=np.mean(y) + 3*np.std(y))
    for peak in peaks:
        start_idx = max(0, peak - 5)
        end_idx = min(len(x), peak + 6)
        mask[start_idx:end_idx] = True
    
    # Find significant changes in signal
    diff = np.abs(np.diff(y))
    significant_changes = np.where(diff > 5*np.std(diff))[0]
    for idx in significant_changes:
        mask[idx:idx+2] = True
    
    # Apply mask
    return x[mask], y[mask]