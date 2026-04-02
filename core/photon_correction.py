"""
Photon Counter Dead-Time Correction

This module implements non-linearity correction for photon counting detectors
that have a dead time period during which they cannot register new photons.

The correction formula is:
    corrected_counts = measured_counts * 1 / (1 - R_measured * T_D)

where:
    R_measured = measured count rate (counts per second)
    T_D = dead time (seconds)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_dead_time_correction(
    signal: np.ndarray,
    time_resolution: float,
    dead_time_ns: float = 43.0
) -> tuple[np.ndarray, dict]:
    """
    Apply dead-time correction to photon counting data.
    
    During the dead time, the detector is blind to incoming photons, causing
    the measured count rate to underestimate the true photon rate. This function
    corrects for this non-linearity.
    
    Parameters
    ----------
    signal : np.ndarray
        Raw measured counts (amplitude data)
    time_resolution : float
        Time per sample in seconds (dwell time)
    dead_time_ns : float
        Detector dead time in nanoseconds (default: 43 ns)
        
    Returns
    -------
    corrected_signal : np.ndarray
        Corrected amplitude data
    correction_info : dict
        Dictionary with correction metadata including:
        - applied: bool
        - dead_time_ns: float
        - max_correction_factor: float
        - mean_correction_factor: float
        - saturated_points: int (points near singularity)
        
    Notes
    -----
    The correction becomes singular when R_measured * T_D approaches 1.
    To avoid numerical issues, we clamp the correction factor at a maximum
    value (default: 10x), which corresponds to ~90% of the theoretical limit.
    
    Examples
    --------
    >>> signal = np.array([100, 200, 300, 400, 500])
    >>> time_res = 1e-4  # 0.1 ms
    >>> corrected, info = apply_dead_time_correction(signal, time_res, 43.0)
    >>> print(f"Max correction: {info['max_correction_factor']:.3f}x")
    """
    if signal is None or len(signal) == 0:
        return signal, {
            'applied': False,
            'reason': 'empty_signal'
        }
    
    # Convert dead time from nanoseconds to seconds
    dead_time_sec = dead_time_ns * 1e-9
    
    # Calculate measured count rate: R = counts / dwell_time
    # time_resolution is the dwell time (seconds per sample)
    count_rate = signal / time_resolution
    
    # Calculate correction factor: 1 / (1 - R * T_D)
    # This approaches infinity as R * T_D approaches 1
    denominator = 1.0 - (count_rate * dead_time_sec)
    
    # Clamp to avoid singularity and numerical instability
    # When denominator < 0.1, correction factor > 10x
    # This corresponds to ~90% of theoretical saturation
    max_correction_factor = 10.0
    min_denominator = 1.0 / max_correction_factor
    
    # Count how many points are near saturation
    saturated_mask = denominator < min_denominator
    saturated_count = int(np.sum(saturated_mask))
    
    if saturated_count > 0:
        logger.warning(
            f"Dead-time correction: {saturated_count} points ({100*saturated_count/len(signal):.1f}%) "
            f"are near saturation (correction factor > {max_correction_factor}x). "
            f"These will be clamped to avoid numerical instability."
        )
    
    # Clamp denominator to avoid division issues
    denominator_clamped = np.maximum(denominator, min_denominator)
    
    # Apply correction
    correction_factor = 1.0 / denominator_clamped
    corrected_signal = signal * correction_factor
    
    # Gather statistics
    correction_info = {
        'applied': True,
        'dead_time_ns': float(dead_time_ns),
        'dead_time_sec': float(dead_time_sec),
        'max_correction_factor': float(np.max(correction_factor)),
        'mean_correction_factor': float(np.mean(correction_factor)),
        'median_correction_factor': float(np.median(correction_factor)),
        'saturated_points': saturated_count,
        'saturation_percentage': float(100 * saturated_count / len(signal)),
        'max_count_rate_hz': float(np.max(count_rate)),
        'mean_count_rate_hz': float(np.mean(count_rate))
    }
    
    logger.info(
        f"Dead-time correction applied: "
        f"mean factor={correction_info['mean_correction_factor']:.3f}x, "
        f"max factor={correction_info['max_correction_factor']:.3f}x, "
        f"mean rate={correction_info['mean_count_rate_hz']:.1f} Hz"
    )
    
    return corrected_signal.astype(np.float32), correction_info


def estimate_saturation_limit(time_resolution: float, dead_time_ns: float = 43.0) -> float:
    """
    Estimate the theoretical saturation count rate for a detector.
    
    Parameters
    ----------
    time_resolution : float
        Time per sample in seconds (dwell time)
    dead_time_ns : float
        Detector dead time in nanoseconds
        
    Returns
    -------
    saturation_rate : float
        Maximum measurable count rate in Hz before complete saturation
        
    Notes
    -----
    Complete saturation occurs when R * T_D = 1, i.e., R = 1 / T_D.
    In practice, the detector becomes unreliable well before this limit.
    """
    dead_time_sec = dead_time_ns * 1e-9
    saturation_rate = 1.0 / dead_time_sec
    return saturation_rate






