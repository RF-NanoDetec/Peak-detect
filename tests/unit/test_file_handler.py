import numpy as np

from core.file_handler import load_single_file


def test_load_single_file_scales_raw_time_and_reads_first_two_columns(tmp_path):
    data_file = tmp_path / "trace.txt"
    data_file.write_text(
        "\n".join(
            [
                "# instrument comment",
                "Raw Time\tCounts\tIgnored",
                "0\t10\t999",
                "",
                "2\t20\t888",
                "4\t30\t777",
            ]
        ),
        encoding="utf-8",
    )

    result = load_single_file(str(data_file), time_resolution=0.5)

    assert np.array_equal(result["time_raw"], np.array([0.0, 2.0, 4.0]))
    assert np.array_equal(result["time"], np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(result["amplitude"], np.array([10.0, 20.0, 30.0]))
    assert result["time_resolution"] == 0.5


def test_load_single_file_preserves_supported_column_names(tmp_path):
    data_file = tmp_path / "trace.txt"
    data_file.write_text(
        "\n".join(
            [
                "Time - Plot 0\tAmplitude - Plot 0\tExtra",
                "0\t1\t100",
                "1\t3\t200",
            ]
        ),
        encoding="utf-8",
    )

    result = load_single_file(str(data_file), time_resolution=1e-4)

    assert np.allclose(result["time"], np.array([0.0, 0.0001]))
    assert np.array_equal(result["amplitude"], np.array([1.0, 3.0]))
