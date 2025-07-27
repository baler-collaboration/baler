# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import abc
import csv
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import zfpy
import blosc2


@dataclass
class BenchmarkResult:
    """A container for the results of a single compression benchmark."""

    name: str
    size_mb: float
    compress_time_sec: float
    decompress_time_sec: float
    rmse: float
    max_err: float
    psnr: float


class Benchmark(abc.ABC):
    """
    Abstract Base Class for a compression benchmark.
    This class defines the template for running a benchmark.
    """

    def __init__(
        self,
        name: str,
        output_dir: str,
        data_original: np.ndarray,
        names_original: np.ndarray,
        verbose: bool = True,
    ):
        self.name = name
        self.output_dir = output_dir
        self.data_original = data_original
        self.names_original = names_original
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> BenchmarkResult:
        """
        Executes the full benchmark process: compress, decompress, and analyze.
        This is the public-facing method to run a benchmark.
        """
        # 1. Compression
        start_compress = time.perf_counter()
        compressed_data = self._compress()
        end_compress = time.perf_counter()
        compress_time = end_compress - start_compress
        compressed_path = self._save_compressed(compressed_data)
        compressed_file_size_bytes = os.path.getsize(compressed_path)

        # 2. Decompression
        start_decompress = time.perf_counter()
        decompressed_data = self._decompress(compressed_data)
        end_decompress = time.perf_counter()
        decompress_time = end_decompress - start_decompress
        self._save_decompressed(decompressed_data)

        # 3. Error Analysis
        metrics = self._analyze_errors(decompressed_data)

        # 4. Print results if verbose
        if self.verbose:
            print(f"\nBenchmarking: {self.name}")
            print(f"  Compression time: {compress_time:.3f} seconds")
            print(
                f"  Compressed size: {compressed_file_size_bytes / (1024 * 1024):.3f} MB"
            )
            print(f"  Decompression time: {decompress_time:.3f} seconds")
            print(
                f"  -> Done. RMSE: {metrics['rmse']:.2e}, Max Error: {metrics['max_err']:.2e}"
            )

        # 5. Create and return result object
        return BenchmarkResult(
            name=self.name,
            size_mb=compressed_file_size_bytes / (1024 * 1024),
            compress_time_sec=compress_time,
            decompress_time_sec=decompress_time,
            **metrics,
        )

    def _analyze_errors(self, decompressed_data: np.ndarray) -> dict:
        """Performs error analysis and returns a dictionary of metrics."""
        if decompressed_data.shape != self.data_original.shape:
            raise ValueError(f"Shape mismatch after decompression for {self.name}!")

        diff = self.data_original.astype(np.float64) - decompressed_data.astype(
            np.float64
        )
        mse = np.mean(diff**2)
        rmse = np.sqrt(mse)
        max_abs_err = np.max(np.abs(diff))

        data_range = np.max(self.data_original) - np.min(self.data_original)
        psnr = (
            20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float("inf")
        )

        return {"rmse": rmse, "max_err": max_abs_err, "psnr": psnr}

    def _save_decompressed(self, decompressed_data: np.ndarray):
        """Saves the decompressed data and names to an NPZ file for inspection."""
        path = os.path.join(self.output_dir, "decompressed.npz")
        np.savez(path, data=decompressed_data, names=self.names_original)
        if self.verbose:
            print(f"  Decompressed file saved to: {path}")

    @abc.abstractmethod
    def _compress(self):
        """Subclasses must implement this. Should return the compressed data."""
        pass

    @abc.abstractmethod
    def _decompress(self, compressed_data) -> np.ndarray:
        """Subclasses must implement this. Should return the decompressed numpy array."""
        pass

    def _save_compressed(self, compressed_data) -> str:
        """Default method to save compressed data as a binary blob."""
        path = os.path.join(self.output_dir, "compressed.bin")
        with open(path, "wb") as f:
            f.write(compressed_data)
        return path


# --- Specific Benchmark Implementations ---


class BalerBenchmark(Benchmark):
    """Benchmark for the main 'baler' autoencoder compression."""

    def __init__(
        self,
        name: str,
        output_path: str,
        compress_func,
        decompress_func,
        data_original,
        names_original,
    ):
        # Baler manages its own output directory, so we pass the project's output_path
        super().__init__(name, output_path, data_original, names_original)
        self.compress_func = compress_func
        self.decompress_func = decompress_func

    def _compress(self):
        # Baler's compression function saves its own file and doesn't return data
        self.compress_func()
        return None  # No data to return

    def _decompress(self, compressed_data):
        # Baler's decompression also works on files, then we load the result
        self.decompress_func()
        decompressed_path = os.path.join(
            self.output_dir, "decompressed_output", "decompressed.npz"
        )
        return np.load(decompressed_path)["data"]

    def _save_compressed(self, compressed_data):
        # This is a no-op because _compress() already saved the file.
        # We just need to return the path for size calculation.
        return os.path.join(self.output_dir, "compressed_output", "compressed.npz")

    def _save_decompressed(self, decompressed_data):
        # This is also a no-op because _decompress() already handled it.
        path = os.path.join(self.output_dir, "decompressed_output", "decompressed.npz")
        if self.verbose:
            print(f"  Decompressed file already saved to: {path}")


class DowncastBenchmark(Benchmark):
    """
    Benchmark for downcasting data to a specified NumPy data type (e.g., float32, float16).
    """

    def __init__(
        self,
        output_dir: str,
        data_original: np.ndarray,
        names_original: np.ndarray,
        target_dtype: np.dtype,
    ):
        """
        Initializes the benchmark for a specific target data type.

        Args:
            output_dir (str): Directory to save benchmark artifacts.
            data_original (np.ndarray): The original, high-precision data.
            names_original (np.ndarray): The names corresponding to the data features.
            target_dtype (np.dtype): The NumPy data type to cast to (e.g., np.float32, np.float16).
        """
        self.target_dtype = target_dtype
        # Automatically generate the name based on the target type
        name = f"Downcast (to {np.dtype(self.target_dtype).name})"
        super().__init__(name, output_dir, data_original, names_original)

    def _compress(self):
        """Performs the downcasting. This is our 'compression' step."""
        return self.data_original.astype(self.target_dtype)

    def _decompress(self, compressed_data):
        """Decompression is a no-op; the compressed data is the final data."""
        return compressed_data

    def _save_compressed(self, compressed_data: np.ndarray) -> str:
        """Override to save as .npz since the 'compressed' data is just a NumPy array."""
        path = os.path.join(self.output_dir, "compressed.npz")
        np.savez(path, data=compressed_data, names=self.names_original)
        return path


class ZFPBenchmark(Benchmark):
    """
    A benchmark class specifically for evaluating the ZFP compression algorithm.
    This class extends the base `Benchmark` to implement compression and decompression
    using the `zfpy` library. It is configured with a dictionary of ZFP-specific
    parameters, such as precision, rate, or tolerance, which are directly passed
    to the `zfpy.compress_numpy` function.
    The name of the benchmark is automatically generated based on the provided
    ZFP parameters to ensure clear identification of the results.
        output_dir (str): The directory where benchmark artifacts (like plots and data) will be saved.
        data_original (np.ndarray): The original, uncompressed data array to be used for the benchmark.
        names_original (np.ndarray): An array of names corresponding to the features in `data_original`.
        zfp_params (dict): A dictionary containing the parameters for ZFP compression. These are passed
                           directly to `zfpy.compress_numpy`. Example: `{'precision': 22}` or `{'rate': 8.0}`.
    Attributes:
        zfp_params (dict): The dictionary of ZFP parameters used for compression.
    """

    def __init__(
        self,
        output_dir: str,
        data_original: np.ndarray,
        names_original: np.ndarray,
        zfp_params: dict,
    ):
        """
        Initializes the benchmark for a specific set of ZFP parameters.

        Args:
            output_dir (str): Directory to save benchmark artifacts.
            data_original (np.ndarray): The original, high-precision data.
            names_original (np.ndarray): The names corresponding to the data features.
            zfp_params (dict): A dictionary of parameters to pass to zfpy.
                               Example: {'precision': 22} or {'rate': 8.0}
        """
        if not zfp_params:
            raise ValueError("zfp_params dictionary cannot be empty for ZFPBenchmark.")

        self.zfp_params = zfp_params

        # Automatically generate a descriptive name from the parameters
        param_str = ", ".join([f"{k}={v}" for k, v in self.zfp_params.items()])
        name = f"ZFP({param_str})"

        super().__init__(name, output_dir, data_original, names_original)

    def _compress(self):
        """Compresses data using the stored ZFP parameters."""
        # The ** operator unpacks the dictionary into keyword arguments
        # e.g., {'rate': 8.0} becomes rate=8.0
        return zfpy.compress_numpy(self.data_original, **self.zfp_params)

    def _decompress(self, compressed_data):
        """Decompression does not require the original parameters."""
        return zfpy.decompress_numpy(compressed_data)


class BloscBenchmark(Benchmark):
    """Benchmark for Blosc2 compression."""

    def __init__(
        self, name: str, output_dir: str, data_original, names_original, cparams: dict
    ):
        super().__init__(name, output_dir, data_original, names_original)
        self.cparams = cparams

    def _compress(self):
        return blosc2.pack_array2(self.data_original, cparams=self.cparams)

    def _decompress(self, compressed_data):
        return blosc2.unpack_array2(compressed_data)


# TODO Implement a benchmark for SZ3 compression


def output_benchmark_results(original_size_mb, all_results, title, output_path, verbose):
    """
    Formats and outputs the results of multiple compression benchmarks.

    This function takes a list of BenchmarkResult objects, sorts them by RMSE,
    and presents them in a formatted table. The summary is printed to the
    console if `verbose` is True, and is always appended to a log file named
    'compression_comparison_results.txt'.

    Args:
        original_size_mb (float): The size of the original, uncompressed data
            in megabytes, used for calculating the compression ratio.
        all_results (list[BenchmarkResult]): A list of result objects from all
            the benchmarks that were run.
        title (str): A title for the summary table, which will be included in
            the header.
        verbose (bool): If True, the summary table is printed to standard
            output.
    """
    # --- Prepare the header for the summary table ---
    header = f"{'Method':<30} | {'Size (MB)':>10} | {'Comp Ratio':>11} | {'RMSE':>10} | {'Max Error':>11} | {'PSNR (dB)':>10} | {'Comp Time(s)':>12} | {'Decomp Time(s)':>14}"

    if verbose:
        # --- Print Final Summary Table ---
        print("\n" + "=" * 150)
        print(
            f"                          COMPRESSION SUMMARY - {title} - Original Size: {original_size_mb:.3f} MB                          "
        )
        print("-" * 150)
        print(header)
        print("-" * 150)

    # --- CSV Logging Setup ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    csv_file_path = f"{output_path}/{timestamp}_compression_comparison_results.csv"
    csv_header = [
        "Original Size (MB)",
        "Method",
        "Size (MB)",
        "Comp Ratio",
        "RMSE",
        "Max Error",
        "PSNR (dB)",
        "Comp Time(s)",
        "Decomp Time(s)",
    ]
    # Check if file exists to decide whether to write the header
    file_exists = os.path.isfile(csv_file_path)

    # --- Write to CSV File ---

    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_header)

        # Sort results by a desired metric, e.g., RMSE, before writing
        sorted_results_for_csv = sorted(all_results, key=lambda r: r.rmse)
        for r in sorted_results_for_csv:
            if original_size_mb > 0 and r.size_mb > 0:
                ratio = original_size_mb / r.size_mb
                ratio_str = f"{ratio:.2f}:1"
            else:
                ratio_str = "N/A"

            row = [
                timestamp,
                title,
                f"{original_size_mb:.3f}",
                r.name,
                f"{r.size_mb:.3f}",
                ratio_str,
                f"{r.rmse:.2e}",
                f"{r.max_err:.2e}",
                f"{r.psnr:.1f}",
                f"{r.compress_time_sec:.3f}",
                f"{r.decompress_time_sec:.3f}",
            ]
            writer.writerow(row)

    # --- Write to Formatted Text Log File ---
    with open("compression_comparison_results.txt", "a") as f:
        f.write("\n" + "=" * 150 + "\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"COMPRESSION SUMMARY - {title} - Original Size: {original_size_mb:.3f} MB\n"
        )
        f.write("-" * 150 + "\n")
        f.write(f"{header}\n")
        f.write("-" * 150 + "\n")

    # Sort results for console/text output
    sorted_results = sorted(all_results, key=lambda r: r.rmse)

    for r in sorted_results:
        # Calculate compression ratio
        if original_size_mb > 0 and r.size_mb > 0:
            ratio = original_size_mb / r.size_mb
            ratio_str = f"{ratio:.2f}:1"
        else:
            ratio_str = "N/A"

        result_string = (
            f"{r.name:<30} | {r.size_mb:>10.3f} | {ratio_str:>11} | {r.rmse:>10.2e} | {r.max_err:>11.2e} | "
            f"{r.psnr:>10.1f} | {r.compress_time_sec:>12.3f} | {r.decompress_time_sec:>14.3f}"
        )

        if verbose:
            # Print each result in a formatted manner
            print(result_string)

        # Write each result to the results tracking file
        with open("compression_comparison_results.txt", "a") as f:
            f.write(result_string + "\n")

    if verbose:
        print("=" * 150 + "\n")
