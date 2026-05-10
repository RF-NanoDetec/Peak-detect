import numpy as np

from core import service_functions


HEADER = "Time - Plot 0 Amplitude - Plot 0"


def test_load_data_from_paths_preserves_preview_and_time_range(monkeypatch):
    samples = {
        "first.txt": {
            "time": np.array([0.0, 1.0], dtype=np.float64),
            "amplitude": np.array([10.0, 20.0], dtype=np.float32),
        },
        "second.txt": {
            "time": np.array([0.0, 1.0], dtype=np.float64),
            "amplitude": np.array([30.0, 40.0], dtype=np.float32),
        },
    }

    def fake_load_single_file(path, timestamps=None, index=0, time_resolution=1e-4):
        sample = samples[path]
        return {
            "index": index,
            "time": sample["time"],
            "amplitude": sample["amplitude"],
            "time_resolution": time_resolution,
        }

    monkeypatch.setattr(service_functions, "load_single_file", fake_load_single_file)
    monkeypatch.setattr(service_functions, "get_tracker", lambda: None)

    result = service_functions.load_data_from_paths(["first.txt", "second.txt"])

    assert np.array_equal(result["time"], np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(result["amplitude"], np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    assert result["loaded_files"] == ["first.txt", "second.txt"]
    assert result["count"] == 2
    assert result["time_range"] == (0.0, 3.0)
    assert result["preview_head"] == "\n".join(
        [
            HEADER,
            "            0                 10",
            "            1                 20",
            "            2                 30",
            "            3                 40",
        ]
    )


def test_load_data_from_paths_returns_header_for_empty_result(monkeypatch):
    def fake_load_single_file(path, timestamps=None, index=0, time_resolution=1e-4):
        return {
            "index": index,
            "time": np.array([], dtype=np.float64),
            "amplitude": np.array([], dtype=np.float32),
            "time_resolution": time_resolution,
        }

    monkeypatch.setattr(service_functions, "load_single_file", fake_load_single_file)
    monkeypatch.setattr(service_functions, "get_tracker", lambda: None)

    result = service_functions.load_data_from_paths(["empty.txt"])

    assert result["time_range"] == (0.0, 0.0)
    assert result["preview_head"] == HEADER


def test_analyze_double_peaks_classifies_consecutive_pairs():
    peaks = np.array([10, 30, 100])
    properties = {
        "prominences": np.array([100.0, 50.0, 50.0]),
        "widths": np.array([5.0, 5.0, 10.0]),
        "left_ips": np.array([8.0, 28.0, 95.0]),
        "right_ips": np.array([13.0, 33.0, 105.0]),
    }

    result = service_functions.analyze_double_peaks_pure(
        peaks,
        properties,
        time_resolution=1e-3,
        min_distance=0.015,
        max_distance=0.030,
        min_amp_ratio=0.4,
        max_amp_ratio=1.2,
        min_width_ratio=0.8,
        max_width_ratio=1.2,
    )

    assert result["total_pairs"] == 2
    assert result["double_peak_count"] == 1
    assert result["peak_pairs"][0]["is_double_peak"] is True
    assert result["peak_pairs"][0]["peak_distance_ms"] == 20.0
    assert result["peak_pairs"][0]["amplitude_ratio"] == 0.5
    assert result["peak_pairs"][1]["is_double_peak"] is False


def test_export_unified_peaks_data_marks_and_filters_double_peaks():
    time_values = np.arange(0, 0.12, 0.001)
    peaks = np.array([10, 30, 100])
    properties = {
        "prominences": np.array([100.0, 50.0, 50.0]),
        "widths": np.array([5.0, 5.0, 10.0]),
        "left_ips": np.array([8.0, 28.0, 95.0]),
        "right_ips": np.array([13.0, 33.0, 105.0]),
    }
    double_peak_analysis = service_functions.analyze_double_peaks_pure(
        peaks,
        properties,
        time_resolution=1e-3,
        min_distance=0.015,
        max_distance=0.030,
        min_amp_ratio=0.4,
        max_amp_ratio=1.2,
        min_width_ratio=0.8,
        max_width_ratio=1.2,
    )

    exported = service_functions.export_unified_peaks_data(
        time_values,
        peaks,
        properties,
        time_resolution=1e-3,
        double_peak_analysis=double_peak_analysis,
        include_double_flags=True,
        filter_double_peaks=True,
        pair_indices=[0],
    )

    peaks_df = exported["peaks_df"]
    double_df = exported["double_df"]

    assert list(peaks_df["Time (s)"]) == [0.01, 0.03]
    assert list(peaks_df["Is Double Peak"]) == [True, True]
    assert list(double_df["Is Double Peak"]) == [True]
    assert list(double_df["Peak Distance (ms)"]) == [20.0]
