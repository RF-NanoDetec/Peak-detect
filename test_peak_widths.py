import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

# Create a simple signal with a clear peak
t = np.linspace(0, 10, 1000)
signal = np.exp(-(t-5)**2) + 0.1 * np.random.randn(1000)

# Find peaks
peaks, properties = find_peaks(signal, height=0.5, distance=100)

# Calculate peak widths and interpolated positions
width_results = peak_widths(signal, peaks, rel_height=0.5)
widths = width_results[0]
width_heights = width_results[1]
left_ips = width_results[2]
right_ips = width_results[3]

# Print detailed information about the first peak
if len(peaks) > 0:
    print("\nPeak Analysis:")
    print(f"Peak position: {peaks[0]}")
    print(f"Peak value: {signal[peaks[0]]}")
    print(f"Peak width: {widths[0]}")
    print(f"Width height: {width_heights[0]}")
    print(f"Left IP: {left_ips[0]}")
    print(f"Right IP: {right_ips[0]}")
    
    # Plot the signal and peak properties
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, 'b-', label='Signal')
    plt.plot(t[peaks], signal[peaks], 'ro', label='Peaks')
    
    # Plot the width measurement points
    for i in range(len(peaks)):
        plt.hlines(width_heights[i], t[int(left_ips[i])], t[int(right_ips[i])], 
                  color='g', linestyle='--', label='Width measurement' if i == 0 else "")
        plt.vlines(t[int(left_ips[i])], 0, width_heights[i], color='r', linestyle=':', 
                  label='Left IP' if i == 0 else "")
        plt.vlines(t[int(right_ips[i])], 0, width_heights[i], color='r', linestyle=':', 
                  label='Right IP' if i == 0 else "")
    
    plt.title('Peak Width Analysis')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show() 