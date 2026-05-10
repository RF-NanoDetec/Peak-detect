import numpy as np

from core.data_utils import decimate_for_plot, validate_peak_params


def test_decimate_preserves_bounds_and_limit():
    """Large inputs are reduced to the target size while keeping endpoints aligned."""
    x = np.arange(0, 100_000, dtype=float)
    y = np.sin(x / 500.0)

    x_dec, y_dec = decimate_for_plot(x, y, max_points=1_000)

    assert len(x_dec) == len(y_dec)
    assert len(x_dec) <= 1_000
    assert x_dec[0] == x[0]
    assert x_dec[-1] == x[-1]


def test_decimate_pass_through_small_arrays():
    """Small inputs should be returned untouched."""
    x = np.linspace(0, 1, 50)
    y = x * 2

    x_dec, y_dec = decimate_for_plot(x, y, max_points=5000)

    assert np.array_equal(x_dec, x)
    assert np.array_equal(y_dec, y)


def test_peak_params_reject_zero_prominence():
    ok, message = validate_peak_params(
        prominence_threshold=0,
        distance=1,
        rel_height=0.8,
        width_ms="0.1,200",
        time_resolution=1e-4,
        prominence_ratio=0,
        max_signal=100,
    )

    assert not ok
    assert "Prominence threshold must be > 0" in message


def test_peak_params_reject_prominence_above_signal_max():
    ok, message = validate_peak_params(
        prominence_threshold=101,
        distance=1,
        rel_height=0.8,
        width_ms="0.1,200",
        time_resolution=1e-4,
        prominence_ratio=0,
        max_signal=100,
    )

    assert not ok
    assert "highest data point" in message


def test_peak_params_reject_width_below_time_resolution():
    ok, message = validate_peak_params(
        prominence_threshold=10,
        distance=1,
        rel_height=0.8,
        width_ms="0.1,200",
        time_resolution=1e-3,
        prominence_ratio=0,
        max_signal=100,
    )

    assert not ok
    assert "time resolution" in message


def test_peak_params_accept_zero_prominence_ratio_as_disabled_filter():
    ok, message = validate_peak_params(
        prominence_threshold=10,
        distance=1,
        rel_height=0.8,
        width_ms="0.1,200",
        time_resolution=1e-4,
        prominence_ratio=0,
        max_signal=100,
    )

    assert ok, message
