"""
Peak detection and analysis module for the Peak Analysis Tool.
Provides functionality for detecting, analyzing, and characterizing peaks in signal data.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor

from .config import config
from .utils.decorators import profile_function

logger = logging.getLogger(__name__)

@dataclass
class Peak:
    """Data structure for individual peak information."""
    index: int
    height: float
    width: float
    prominence: float
    left_base: int
    right_base: int
    area: float
    center_mass: float
    timestamp: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert peak information to dictionary."""
        return {
            'index': self.index,
            'height': self.height,
            'width': self.width,
            'prominence': self.prominence,
            'left_base': self.left_base,
            'right_base': self.right_base,
            'area': self.area,
            'center_mass': self.center_mass,
            'timestamp': self.timestamp
        }

class PeakDetector:
    """Class for peak detection and analysis."""
    
    def __init__(self):
        """Initialize peak detector with default parameters."""
        self.peaks: List[Peak] = []
        self.signal: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.sampling_rate: Optional[float] = None
    
    @profile_function
    def find_peaks_with_window(
        self,
        signal: np.ndarray,
        height: float,
        distance: int,
        width: Tuple[int, int],
        prominence: Optional[float] = None,
        rel_height: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Find peaks in signal using specified parameters.
        
        Args:
            signal: Input signal array
            height: Minimum peak height
            distance: Minimum distance between peaks
            width: Tuple of (min_width, max_width)
            prominence: Minimum peak prominence
            rel_height: Relative height for width calculation
            
        Returns:
            Tuple of peak indices and properties dictionary
        """
        try:
            logger.info("Starting peak detection with parameters:")
            logger.info(f"Height: {height}, Distance: {distance}, Width: {width}")
            
            # Find peaks
            peaks, properties = find_peaks(
                signal,
                height=height,
                distance=distance,
                width=width,
                prominence=prominence,
                rel_height=rel_height
            )
            
            logger.info(f"Found {len(peaks)} peaks")
            return peaks, properties
            
        except Exception as e:
            logger.error(f"Error in peak detection: {str(e)}")
            raise
    
    @profile_function
    def analyze_peaks(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        peaks: np.ndarray,
        properties: Dict,
        sampling_rate: float
    ) -> List[Peak]:
        """
        Analyze detected peaks and calculate their properties.
        
        Args:
            signal: Input signal array
            time: Time array
            peaks: Array of peak indices
            properties: Peak properties dictionary
            sampling_rate: Sampling rate of the signal
            
        Returns:
            List of Peak objects with calculated properties
        """
        analyzed_peaks = []
        
        try:
            for i, peak_idx in enumerate(peaks):
                # Calculate peak area
                left_idx = int(properties['left_bases'][i])
                right_idx = int(properties['right_bases'][i])
                peak_signal = signal[left_idx:right_idx+1]
                baseline = min(signal[left_idx], signal[right_idx])
                area = np.trapz(peak_signal - baseline, dx=1/sampling_rate)
                
                # Calculate center of mass
                mass_signal = peak_signal - baseline
                center_mass = np.sum(time[left_idx:right_idx+1] * mass_signal) / np.sum(mass_signal)
                
                # Create Peak object
                peak = Peak(
                    index=peak_idx,
                    height=properties['peak_heights'][i],
                    width=properties['widths'][i] / sampling_rate,
                    prominence=properties['prominences'][i],
                    left_base=left_idx,
                    right_base=right_idx,
                    area=area,
                    center_mass=center_mass
                )
                
                analyzed_peaks.append(peak)
                
            logger.info(f"Analyzed {len(analyzed_peaks)} peaks")
            return analyzed_peaks
            
        except Exception as e:
            logger.error(f"Error in peak analysis: {str(e)}")
            raise
    
    def estimate_noise_threshold(self, signal: np.ndarray) -> float:
        """
        Estimate noise threshold for peak detection.
        
        Args:
            signal: Input signal array
            
        Returns:
            Estimated threshold value
        """
        # Calculate signal statistics
        signal_std = np.std(signal)
        signal_mean = np.mean(signal)
        
        # Use configurable number of standard deviations
        threshold = signal_mean + (config.analysis.NOISE_THRESHOLD * signal_std)
        
        logger.info(f"Estimated noise threshold: {threshold:.2f}")
        return threshold
    
    def detect_peak_clusters(
        self,
        peaks: List[Peak],
        max_gap: float
    ) -> List[List[Peak]]:
        """
        Detect clusters of peaks based on time separation.
        
        Args:
            peaks: List of Peak objects
            max_gap: Maximum time gap between peaks in cluster
            
        Returns:
            List of peak clusters
        """
        if not peaks:
            return []
        
        # Sort peaks by index
        sorted_peaks = sorted(peaks, key=lambda x: x.index)
        clusters = [[sorted_peaks[0]]]
        
        for peak in sorted_peaks[1:]:
            if (peak.index - clusters[-1][-1].index) / self.sampling_rate <= max_gap:
                clusters[-1].append(peak)
            else:
                clusters.append([peak])
        
        logger.info(f"Found {len(clusters)} peak clusters")
        return clusters
    
    def calculate_peak_statistics(self) -> Dict[str, float]:
        """
        Calculate statistical measures for detected peaks.
        
        Returns:
            Dictionary of peak statistics
        """
        if not self.peaks:
            return {}
        
        heights = [p.height for p in self.peaks]
        widths = [p.width for p in self.peaks]
        areas = [p.area for p in self.peaks]
        
        stats = {
            'peak_count': len(self.peaks),
            'mean_height': np.mean(heights),
            'std_height': np.std(heights),
            'mean_width': np.mean(widths),
            'std_width': np.std(widths),
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'peak_rate': len(self.peaks) / (self.time[-1] - self.time[0])
        }
        
        return stats
    
    @profile_function
    def process_signal(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sampling_rate: float,
        **peak_params
    ) -> Tuple[List[Peak], Dict[str, float]]:
        """
        Complete signal processing pipeline for peak detection.
        
        Args:
            signal: Input signal array
            time: Time array
            sampling_rate: Sampling rate of the signal
            **peak_params: Peak detection parameters
            
        Returns:
            Tuple of peak list and statistics dictionary
        """
        try:
            self.signal = signal
            self.time = time
            self.sampling_rate = sampling_rate
            
            # Detect peaks
            peaks, properties = self.find_peaks_with_window(
                signal=signal,
                **peak_params
            )
            
            # Analyze peaks
            self.peaks = self.analyze_peaks(
                signal=signal,
                time=time,
                peaks=peaks,
                properties=properties,
                sampling_rate=sampling_rate
            )
            
            # Calculate statistics
            stats = self.calculate_peak_statistics()
            
            return self.peaks, stats
            
        except Exception as e:
            logger.error(f"Error in signal processing: {str(e)}")
            raise
    
    def export_peaks(self, filepath: str) -> None:
        """
        Export peak information to CSV file.
        
        Args:
            filepath: Path to save the file
        """
        try:
            import pandas as pd
            
            # Convert peaks to dictionaries
            peak_dicts = [peak.to_dict() for peak in self.peaks]
            
            # Create DataFrame and save
            df = pd.DataFrame(peak_dicts)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Peak data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting peaks: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = PeakDetector()
    
    # Example signal
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, len(t))
    
    try:
        # Process signal
        peaks, stats = detector.process_signal(
            signal=signal,
            time=t,
            sampling_rate=100,
            height=0.5,
            distance=20,
            width=(1, 50)
        )
        
        print("Peak statistics:", stats)
        
        # Export results
        detector.export_peaks('peaks.csv')
        
    except Exception as e:
        print(f"Error: {str(e)}")
