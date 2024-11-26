"""
Data processing module for the Peak Analysis Tool.
Handles data loading, processing, and management.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .config import config
from .utils.decorators import profile_function

logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Data structure for individual measurements."""
    time: float
    amplitude: float
    timestamp: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'time': self.time,
            'amplitude': self.amplitude,
            'timestamp': self.timestamp
        }

@dataclass
class ProcessedData:
    """Container for processed measurement data."""
    time: np.ndarray
    amplitude: np.ndarray
    filtered_signal: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None
    peak_properties: Optional[dict] = None
    sampling_rate: Optional[float] = None
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range of the data."""
        return self.time[0], self.time[-1]
    
    def get_amplitude_range(self) -> Tuple[float, float]:
        """Get the amplitude range of the data."""
        return np.min(self.amplitude), np.max(self.amplitude)

class DataProcessor:
    """Handles data loading and processing operations."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.data: Optional[ProcessedData] = None
        self.file_paths: List[str] = []
        self.timestamps: List[str] = []
        self._sampling_rate: Optional[float] = None
    
    @profile_function
    def load_single_file(self, file_path: str, timestamp: Optional[str] = None) -> Dict:
        """
        Load data from a single file.
        
        Args:
            file_path: Path to the data file
            timestamp: Optional timestamp for the file
            
        Returns:
            Dictionary containing the loaded data
        """
        try:
            logger.info(f"Loading file: {file_path}")
            
            # Read the file
            df = pd.read_csv(
                file_path,
                delimiter=config.file.DEFAULT_DELIMITER,
                encoding=config.file.ENCODING
            )
            
            # Clean column names
            df.columns = [col.strip() for col in df.columns]
            
            # Check for required columns
            if config.file.TIME_COLUMN in df.columns:
                time_col = config.file.TIME_COLUMN
                amp_col = config.file.AMPLITUDE_COLUMN
            else:
                # Assume first two columns are time and amplitude
                df.columns = [config.file.TIME_COLUMN, config.file.AMPLITUDE_COLUMN]
                time_col = config.file.TIME_COLUMN
                amp_col = config.file.AMPLITUDE_COLUMN
            
            return {
                'time': df[time_col].values,
                'amplitude': df[amp_col].values,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    @profile_function
    def load_files(self, file_paths: List[str], timestamps: Optional[List[str]] = None) -> None:
        """
        Load multiple files in parallel.
        
        Args:
            file_paths: List of paths to data files
            timestamps: Optional list of timestamps for batch processing
        """
        self.file_paths = file_paths
        self.timestamps = timestamps or [None] * len(file_paths)
        
        try:
            results = []
            with ThreadPoolExecutor(max_workers=config.system.MAX_WORKERS) as executor:
                future_to_file = {
                    executor.submit(
                        self.load_single_file, 
                        file_path, 
                        timestamp
                    ): i 
                    for i, (file_path, timestamp) in enumerate(zip(file_paths, self.timestamps))
                }
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append((future_to_file[future], result))
            
            # Sort results by index
            results.sort(key=lambda x: x[0])
            
            # Combine results
            self._combine_results([r[1] for r in results])
            
        except Exception as e:
            logger.error(f"Error in batch loading: {str(e)}")
            raise
    
    def _combine_results(self, results: List[Dict]) -> None:
        """
        Combine results from multiple files.
        
        Args:
            results: List of dictionaries containing file data
        """
        all_times = []
        all_amplitudes = []
        
        for i, result in enumerate(results):
            if i == 0:
                all_times.append(result['time'])
            else:
                # Add offset based on previous data
                time_offset = all_times[-1][-1] + (result['time'][1] - result['time'][0])
                all_times.append(result['time'] + time_offset)
            
            all_amplitudes.append(result['amplitude'])
        
        # Combine all data
        combined_time = np.concatenate(all_times)
        combined_amplitude = np.concatenate(all_amplitudes)
        
        # Calculate sampling rate
        self._sampling_rate = 1 / np.mean(np.diff(combined_time))
        
        # Create ProcessedData object
        self.data = ProcessedData(
            time=combined_time,
            amplitude=combined_amplitude,
            sampling_rate=self._sampling_rate
        )
        
        logger.info(f"Combined {len(results)} files, total points: {len(combined_time)}")
    
    def decimate_for_plotting(self, max_points: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decimate data for plotting to reduce memory usage.
        
        Args:
            max_points: Maximum number of points to plot
            
        Returns:
            Tuple of decimated time and amplitude arrays
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        if max_points is None:
            max_points = config.plot.MAX_PLOT_POINTS
        
        if len(self.data.time) > max_points:
            # Calculate stride for decimation
            stride = len(self.data.time) // max_points
            
            # Decimate data
            time_decimated = self.data.time[::stride]
            amplitude_decimated = self.data.amplitude[::stride]
            
            logger.info(f"Decimated {len(self.data.time)} points to {len(time_decimated)}")
            
            return time_decimated, amplitude_decimated
        
        return self.data.time, self.data.amplitude
    
    def get_chunk(self, start_idx: int, chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a chunk of data for processing.
        
        Args:
            start_idx: Starting index
            chunk_size: Size of the chunk
            
        Returns:
            Tuple of time and amplitude arrays for the chunk
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        end_idx = min(start_idx + chunk_size, len(self.data.time))
        
        return (
            self.data.time[start_idx:end_idx],
            self.data.amplitude[start_idx:end_idx]
        )
    
    def calculate_statistics(self) -> Dict[str, float]:
        """
        Calculate basic statistics of the data.
        
        Returns:
            Dictionary containing statistical measures
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        stats = {
            'mean': np.mean(self.data.amplitude),
            'std': np.std(self.data.amplitude),
            'min': np.min(self.data.amplitude),
            'max': np.max(self.data.amplitude),
            'median': np.median(self.data.amplitude),
            'sampling_rate': self._sampling_rate
        }
        
        if self.data.peaks is not None:
            stats.update({
                'peak_count': len(self.data.peaks),
                'mean_peak_height': np.mean(self.data.peak_properties['peak_heights']),
                'mean_peak_width': np.mean(self.data.peak_properties['widths'])
            })
        
        return stats
    
    def export_data(self, filepath: str, include_processed: bool = True) -> None:
        """
        Export data to CSV file.
        
        Args:
            filepath: Path to save the file
            include_processed: Whether to include processed data
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        try:
            data_dict = {
                'Time': self.data.time,
                'Amplitude': self.data.amplitude
            }
            
            if include_processed and self.data.filtered_signal is not None:
                data_dict['Filtered_Signal'] = self.data.filtered_signal
            
            if include_processed and self.data.peaks is not None:
                peak_signal = np.zeros_like(self.data.amplitude)
                peak_signal[self.data.peaks] = self.data.amplitude[self.data.peaks]
                data_dict['Peaks'] = peak_signal
            
            df = pd.DataFrame(data_dict)
            df.to_csv(
                filepath,
                index=False,
                sep=config.file.EXPORT_DELIMITER,
                encoding=config.file.EXPORT_ENCODING
            )
            
            logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def clear_data(self) -> None:
        """Clear all loaded data."""
        self.data = None
        self.file_paths = []
        self.timestamps = []
        self._sampling_rate = None
        logger.info("Data cleared")

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create processor instance
    processor = DataProcessor()
    
    # Example file paths
    files = ['data1.txt', 'data2.txt']
    timestamps = ['00:00', '01:00']
    
    try:
        # Load files
        processor.load_files(files, timestamps)
        
        # Get statistics
        stats = processor.calculate_statistics()
        print("Data statistics:", stats)
        
        # Export data
        processor.export_data('processed_data.csv')
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        processor.clear_data()
