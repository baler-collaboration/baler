import os
import shutil
import numpy as np
import pytest
from baler.modules.compare import Benchmark, BenchmarkResult


# A concrete implementation of the abstract Benchmark class for testing purposes
class MockBenchmark(Benchmark):
    """A mock benchmark for testing the base Benchmark class functionality."""

    def __init__(self, *args, **kwargs):
        self.compress_called = False
        self.decompress_called = False
        self.noise_level = kwargs.pop("noise_level", 0.1)
        super().__init__(*args, **kwargs)

    def _compress(self):
        """Mock compression: returns the data as a byte string."""
        self.compress_called = True
        return self.data_original.tobytes()

    def _decompress(self, compressed_data) -> np.ndarray:
        """Mock decompression: converts bytes back to array and adds some noise."""
        self.decompress_called = True
        decompressed = np.frombuffer(compressed_data, dtype=self.data_original.dtype)
        decompressed = decompressed.reshape(self.data_original.shape)
        # Add a small, predictable error for analysis
        return decompressed + self.noise_level


@pytest.fixture
def benchmark_setup(tmp_path):
    """Provides a setup for benchmark tests, including a temp directory and data."""
    output_dir = tmp_path / "benchmark_output"
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    names = np.array([f"var_{i}" for i in range(10)])

    # Use a specific noise level for predictable results
    noise = 0.1
    benchmark = MockBenchmark(
        name="MockTest",
        output_dir=str(output_dir),
        data_original=data,
        names_original=names,
        verbose=False,
        noise_level=noise,
    )

    yield benchmark, data, names, output_dir, noise

    # Teardown is handled by tmp_path fixture


def test_benchmark_initialization(benchmark_setup):
    """Tests if the Benchmark class initializes correctly."""
    benchmark, data, names, output_dir, _ = benchmark_setup
    assert benchmark.name == "MockTest"
    assert benchmark.output_dir == str(output_dir)
    np.testing.assert_array_equal(benchmark.data_original, data)
    np.testing.assert_array_equal(benchmark.names_original, names)
    assert os.path.exists(output_dir)


def test_analyze_errors(benchmark_setup):
    """Tests the _analyze_errors method with various scenarios."""
    benchmark, original_data, _, _, _ = benchmark_setup

    # Scenario 1: No error
    metrics_no_error = benchmark._analyze_errors(original_data)
    assert metrics_no_error["rmse"] == 0.0
    assert metrics_no_error["max_err"] == 0.0
    assert metrics_no_error["psnr"] == float("inf")

    # Scenario 2: With a known, constant error
    decompressed_with_error = original_data + 0.5
    metrics_with_error = benchmark._analyze_errors(decompressed_with_error)
    assert np.isclose(metrics_with_error["rmse"], 0.5)
    assert np.isclose(metrics_with_error["max_err"], 0.5)

    # Scenario 3: Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch after decompression"):
        benchmark._analyze_errors(original_data.flatten())


def test_save_decompressed(benchmark_setup):
    """Tests if the decompressed file is saved correctly."""
    benchmark, _, names, output_dir, _ = benchmark_setup
    decompressed_data = np.random.rand(10, 10)

    benchmark._save_decompressed(decompressed_data)

    saved_path = output_dir / "decompressed.npz"
    assert os.path.exists(saved_path)

    loaded_file = np.load(saved_path)
    np.testing.assert_array_equal(loaded_file["data"], decompressed_data)
    np.testing.assert_array_equal(loaded_file["names"], names)


def test_save_compressed(benchmark_setup):
    """Tests the default binary blob saving for compressed data."""
    benchmark, _, _, output_dir, _ = benchmark_setup
    compressed_content = b"\x01\x02\x03\x04\x05"

    saved_path_str = benchmark._save_compressed(compressed_content)
    saved_path = output_dir / "compressed.bin"

    assert saved_path_str == str(saved_path)
    assert os.path.exists(saved_path)

    with open(saved_path, "rb") as f:
        read_content = f.read()

    assert read_content == compressed_content


def test_run_full_workflow(benchmark_setup):
    """Tests the entire run() method from compression to result generation."""
    benchmark, original_data, _, _, noise = benchmark_setup

    result = benchmark.run()

    # Check if mock methods were called
    assert benchmark.compress_called
    assert benchmark.decompress_called

    # Check the result object
    assert isinstance(result, BenchmarkResult)
    assert result.name == "MockTest"
    assert result.compress_time_sec > 0
    assert result.decompress_time_sec > 0
    assert result.size_mb > 0

    # Check error metrics based on the known noise
    assert np.isclose(result.rmse, noise)
    assert np.isclose(result.max_err, noise)
    assert result.psnr > 0

    # Check if files were created
    assert os.path.exists(os.path.join(benchmark.output_dir, "compressed.bin"))
    assert os.path.exists(os.path.join(benchmark.output_dir, "decompressed.npz"))
