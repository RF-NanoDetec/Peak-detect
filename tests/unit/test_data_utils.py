import numpy as np

from core.data_utils import decimate_for_plot


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
