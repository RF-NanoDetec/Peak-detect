import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.peak_detection import PeakDetector
from core.peak_analysis_utils import apply_butterworth_filter
from core.photon_correction import apply_dead_time_correction

# Setup style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300
})

OUTPUT_DIR = os.path.join('docs', 'figures', 'generated')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_signal(n_points=1000, n_peaks=5, noise_level=0.1):
    """Generate a synthetic signal with Gaussian peaks."""
    t = np.linspace(0, 10, n_points)
    signal = np.zeros_like(t) + 0.5  # Baseline
    
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, n_points)
    
    # Add peaks
    peak_locs = np.linspace(1, 9, n_peaks)
    for loc in peak_locs:
        # Random amplitude between 2 and 5
        amp = np.random.uniform(2, 5)
        # Gaussian peak
        signal += amp * np.exp(-(t - loc)**2 / (2 * 0.1**2))
        
    return t, signal + noise

def plot_single_peak_analysis():
    """Generate plot showing raw vs filtered signal and peak detection."""
    print("Generating single_peak_analysis.png...")
    
    t, raw_signal = generate_synthetic_signal(n_points=2000, n_peaks=3, noise_level=0.2)
    
    # Apply filter
    filtered_signal = apply_butterworth_filter(2, 0.1, 'lowpass', 100, raw_signal)
    
    # Detect peaks
    detector = PeakDetector()
    peaks, props = detector.detect_peaks(filtered_signal, t, height_lim=1.0, distance=50, prominence_ratio=0.5)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot 1: Raw vs Filtered
    ax1.plot(t, raw_signal, 'gray', alpha=0.5, label='Raw Signal')
    ax1.plot(t, filtered_signal, 'b-', linewidth=1.5, label='Filtered (Butterworth)')
    ax1.set_ylabel('Amplitude (V)')
    ax1.set_title('Signal Pre-processing')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Peak Detection
    ax2.plot(t, filtered_signal, 'b-', label='Signal')
    ax2.plot(t[peaks], filtered_signal[peaks], 'rx', markersize=10, label='Detected Peaks')
    
    # Annotate width
    for i, peak in enumerate(peaks):
        width = props['widths'][i]
        # Convert width from samples to time units (approx)
        # width in samples needs to be mapped to time domain
        # Here we just visualize the width height
        h = props['width_heights'][i]
        l = props['left_ips'][i] * (t[1] - t[0])
        r = props['right_ips'][i] * (t[1] - t[0])
        ax2.hlines(h, l, r, color='g', linestyle='--', alpha=0.8)
        if i == 0:
            ax2.text(t[peak], h + 0.2, 'FWHM', color='g', ha='center')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (V)')
    ax2.set_title('Peak Detection & Characterization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'single_peak_analysis.png'))
    plt.close()

def plot_double_peak_resolution():
    """Generate plot showing double peak resolution."""
    print("Generating double_peak_resolution.png...")
    
    t = np.linspace(0, 5, 500)
    # Create two overlapping peaks
    p1 = 3 * np.exp(-(t - 2.4)**2 / (2 * 0.08**2))
    p2 = 2.5 * np.exp(-(t - 2.6)**2 / (2 * 0.08**2))
    signal = p1 + p2 + np.random.normal(0, 0.05, len(t))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t, signal, 'k-', label='Combined Signal')
    ax.plot(t, p1, 'b--', alpha=0.5, label='Component 1')
    ax.plot(t, p2, 'g--', alpha=0.5, label='Component 2')
    
    # Find peaks
    peaks, _ = find_peaks(signal, distance=10, prominence=0.5)
    ax.plot(t[peaks], signal[peaks], 'ro', label='Identified Maxima')
    
    ax.set_title('Double Peak Resolution')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'double_peak_resolution.png'))
    plt.close()

def plot_dead_time_correction():
    """Generate plot for dead time correction."""
    print("Generating dead_time_correction.png...")
    
    # Simulate high count rate signal
    t = np.linspace(0, 10, 1000)
    true_rate = 50e6 * np.exp(-(t - 5)**2 / 2)  # Peak at 50 MHz
    dead_time = 43e-9  # 43ns
    
    # Measured rate formula: R_m = R_t / (1 + R_t * tau)
    measured_rate = true_rate / (1 + true_rate * dead_time)
    
    # Convert to counts per bin (assuming 10ms bins for visualization scale)
    time_res = 1e-4 # 0.1 ms
    # Simulate the counts we would see in the buffer
    measured_counts = measured_rate * time_res
    
    # Apply correction function
    corrected_counts, info = apply_dead_time_correction(measured_counts, time_res, dead_time_ns=43.0)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.plot(t, measured_counts, 'r-', label='Measured (Saturation)', linewidth=2)
    ax1.plot(t, corrected_counts, 'b--', label='Corrected (Reconstructed)', linewidth=2)
    
    # Plot theoretical limit
    saturation_limit = (1/dead_time) * time_res
    ax1.axhline(saturation_limit, color='k', linestyle=':', label='Theoretical Saturation Limit')
    
    ax1.set_title(f'Dead-Time Correction ($\tau_D = {dead_time*1e9:.0f}$ ns)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Counts / 0.1ms')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dead_time_correction.png'))
    plt.close()

def plot_baseline_noise():
    """Generate plot showing baseline noise estimation."""
    print("Generating baseline_noise.png...")
    
    t = np.linspace(0, 10, 1000)
    # Signal with varying noise
    signal = np.zeros_like(t)
    signal[300:350] += 5  # A peak
    
    # Background noise
    noise = np.random.normal(0, 0.5, len(t))
    noisy_signal = signal + noise
    
    # Calculate MAD
    median = np.median(noisy_signal)
    mad = np.median(np.abs(noisy_signal - median))
    sigma_mad = 1.4826 * mad
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t, noisy_signal, 'gray', alpha=0.6, label='Noisy Signal')
    ax.axhline(median, color='b', linestyle='-', label='Median Baseline')
    ax.axhline(median + 3*sigma_mad, color='r', linestyle='--', label='+3$\sigma_{MAD}$')
    ax.axhline(median - 3*sigma_mad, color='r', linestyle='--', label='-3$\sigma_{MAD}$')
    
    ax.fill_between(t, median - sigma_mad, median + sigma_mad, color='b', alpha=0.1, label='$\pm 1 \sigma_{MAD}$ Region')
    
    ax.set_title('Robust Baseline Noise Estimation (MAD)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_noise.png'))
    plt.close()

if __name__ == "__main__":
    print("Starting documentation image generation...")
    plot_single_peak_analysis()
    plot_double_peak_resolution()
    plot_dead_time_correction()
    plot_baseline_noise()
    print(f"All images generated in {OUTPUT_DIR}")






