import numpy as np

from core.peak_detection import PeakDetector


def _gaussian_signal(time_resolution, duration=0.2, center=0.1, fwhm=0.02):
    time = np.arange(0, duration, time_resolution)
    sigma = fwhm / 2.354820045
    signal = 100.0 * np.exp(-0.5 * ((time - center) / sigma) ** 2)
    return time, signal


def test_peak_width_stays_in_milliseconds_across_time_resolutions():
    measured_widths_ms = []

    for time_resolution in (1e-3, 5e-4):
        time, signal = _gaussian_signal(time_resolution)
        detector = PeakDetector()

        peaks, properties = detector.detect_peaks(
            signal=signal,
            time_values=time,
            height_lim=10,
            distance=int(0.05 / time_resolution),
            prominence_ratio=0,
            rel_height=0.5,
            width_range=[10, 30],
            time_resolution=time_resolution,
        )

        assert len(peaks) == 1
        measured_widths_ms.append(properties["widths"][0] * time_resolution * 1000)

    assert np.allclose(measured_widths_ms, [20.0, 20.0], atol=0.5)
