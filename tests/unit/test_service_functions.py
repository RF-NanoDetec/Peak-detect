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
