"""
Unit tests for photon counter dead-time correction.
"""

import pytest
import numpy as np
from core.photon_correction import apply_dead_time_correction, estimate_saturation_limit


class TestDeadTimeCorrection:
    """Test suite for dead-time correction functionality."""
    
    def test_basic_correction(self):
        """Test basic correction with typical values."""
        # Create a simple signal with known values
        signal = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
        time_resolution = 1e-4  # 0.1 ms
        dead_time_ns = 43.0
        
        corrected, info = apply_dead_time_correction(signal, time_resolution, dead_time_ns)
        
        # Correction should increase the signal
        assert np.all(corrected >= signal), "Corrected signal should be >= original"
        
        # Check metadata
        assert info['applied'] is True
        assert info['dead_time_ns'] == 43.0
        assert info['max_correction_factor'] > 1.0
        assert info['mean_correction_factor'] > 1.0
        
    def test_zero_signal(self):
        """Test that zero signal remains zero."""
        signal = np.zeros(100, dtype=np.float32)
        time_resolution = 1e-4
        dead_time_ns = 43.0
        
        corrected, info = apply_dead_time_correction(signal, time_resolution, dead_time_ns)
        
        # Zero signal should remain zero (correction factor = 1)
        assert np.allclose(corrected, signal), "Zero signal should remain zero"
        assert info['mean_correction_factor'] == 1.0
        
    def test_low_count_rate(self):
        """Test correction at low count rates (minimal correction needed)."""
        # Low counts: 10 counts per 0.1 ms = 100 kHz
        signal = np.full(100, 10.0, dtype=np.float32)
        time_resolution = 1e-4
        dead_time_ns = 43.0
        
        corrected, info = apply_dead_time_correction(signal, time_resolution, dead_time_ns)
        
        # At low rates, correction should be small
        correction_factor = corrected[0] / signal[0]
        assert 1.0 < correction_factor < 1.01, "Low rate correction should be minimal"
        
    def test_high_count_rate(self):
        """Test correction at high count rates."""
        # High counts: 5000 counts per 0.1 ms = 50 MHz
        signal = np.full(100, 5000.0, dtype=np.float32)
        time_resolution = 1e-4
        dead_time_ns = 43.0
        
        corrected, info = apply_dead_time_correction(signal, time_resolution, dead_time_ns)
        
        # At high rates, correction should be significant
        correction_factor = corrected[0] / signal[0]
        assert correction_factor > 1.2, "High rate correction should be significant"
        
    def test_saturation_clamping(self):
        """Test that extreme values are clamped to avoid singularity."""
        # Create signal that would cause saturation
        # Dead time = 43 ns = 43e-9 s
        # Saturation at R = 1/T_D ≈ 23 MHz
        # With time_res = 1e-4, this is ~2300 counts per sample
        signal = np.array([10000.0, 20000.0, 30000.0], dtype=np.float32)
        time_resolution = 1e-4
        dead_time_ns = 43.0
        
        corrected, info = apply_dead_time_correction(signal, time_resolution, dead_time_ns)
        
        # Should have saturated points
        assert info['saturated_points'] > 0, "Should detect saturated points"
        
        # Correction factor should be clamped at max (10x)
        assert info['max_correction_factor'] <= 10.0, "Should clamp at max factor"
        
    def test_empty_signal(self):
        """Test handling of empty signal."""
        signal = np.array([], dtype=np.float32)
        time_resolution = 1e-4
        dead_time_ns = 43.0
        
        corrected, info = apply_dead_time_correction(signal, time_resolution, dead_time_ns)
        
        assert len(corrected) == 0
        assert info['applied'] is False
        
    def test_different_dead_times(self):
        """Test correction with different dead time values."""
        signal = np.full(100, 1000.0, dtype=np.float32)
        time_resolution = 1e-4
        
        # Test with different dead times
        dead_times = [10.0, 43.0, 100.0]  # ns
        correction_factors = []
        
        for dt in dead_times:
            corrected, info = apply_dead_time_correction(signal, time_resolution, dt)
            correction_factors.append(info['mean_correction_factor'])
        
        # Longer dead time should require larger correction
        assert correction_factors[0] < correction_factors[1] < correction_factors[2], \
            "Correction should increase with dead time"
    
    def test_correction_formula(self):
        """Test that correction formula is applied correctly."""
        # Use simple values for manual verification
        counts = 1000.0
        time_resolution = 1e-4  # 0.1 ms
        dead_time_ns = 43.0
        
        signal = np.array([counts], dtype=np.float32)
        corrected, info = apply_dead_time_correction(signal, time_resolution, dead_time_ns)
        
        # Manual calculation
        dead_time_sec = 43e-9
        count_rate = counts / time_resolution  # Hz
        expected_factor = 1.0 / (1.0 - count_rate * dead_time_sec)
        expected_corrected = counts * expected_factor
        
        # Should match within floating point precision
        assert np.isclose(corrected[0], expected_corrected, rtol=1e-5), \
            f"Expected {expected_corrected}, got {corrected[0]}"


class TestSaturationLimit:
    """Test saturation limit estimation."""
    
    def test_saturation_calculation(self):
        """Test theoretical saturation limit calculation."""
        dead_time_ns = 43.0
        time_resolution = 1e-4
        
        saturation_rate = estimate_saturation_limit(time_resolution, dead_time_ns)
        
        # For 43 ns dead time, saturation is at 1/(43e-9) ≈ 23.3 MHz
        expected = 1.0 / (43e-9)
        assert np.isclose(saturation_rate, expected, rtol=1e-5)
        
    def test_different_dead_times_saturation(self):
        """Test that shorter dead times allow higher saturation rates."""
        time_resolution = 1e-4
        
        rate_10ns = estimate_saturation_limit(time_resolution, 10.0)
        rate_100ns = estimate_saturation_limit(time_resolution, 100.0)
        
        # Shorter dead time = higher saturation rate
        assert rate_10ns > rate_100ns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])




