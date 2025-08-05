# Copyright 2022-2025 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os
import time
from datetime import datetime
from time import perf_counter
import contextlib
import psutil
import cpuinfo
import shutil

# Attempt to use pynvml for NVIDIA GPU tracking
try:
    import pynvml

    pynvml.nvmlInit()
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False
    if shutil.which("nvidia-smi") is not None:
        print("\n" + "=" * 80)
        print("WARNING: NVIDIA GPU detected, but 'pynvml' is not installed.")
        print("To enable GPU tracking, please install the optional 'gpu' dependencies:")
        print("\n    poetry install --with gpu\n")
        print("=" * 80 + "\n")
except pynvml.NVMLError:
    # if pynvml is installed but driver communication fails
    NVIDIA_SMI_AVAILABLE = False
    print(
        "\nWARNING: 'pynvml' is installed, but could not connect to the NVIDIA driver."
    )
    print("GPU tracking will be disabled. Check your driver installation.\n")


class GreenCodeTracker:
    """
    A class to track function run times and system specs, saving the data to a CSV file.

    This class can be used as a context manager to easily time blocks of code.
    It records hardware specifications like CPU and GPU models, core counts, and
    available memory, along with the runtime of the specified code block.

    Attributes:
        file_name (str): The name of the CSV file where tracking data is stored.
        headers (list): The column headers for the CSV file.
        system_specs (dict): A dictionary containing details about the system's hardware.
    """

    def __init__(self, file_path="green_code_tracking.csv"):
        """
        Initializes the GreenCodeTracker.

        Args:

        """
        self.file_name = file_path
        self.headers = [
            "Timestamp",
            "Runtime(format HH:MM:SS)",
            "Title",
            "CPUmodel",
            "GPU model",
            "Number of CPU cores",
            "Number of GPU cores",
            "Memory available (GB)",
        ]
        self.system_specs = self._get_system_specs()
        self._ensure_header()

    def _get_system_specs(self):
        """
        Gathers system hardware specifications.

        Collects information about the CPU, GPU (if available), and memory.

        Returns:
            dict: A dictionary containing system specifications.
        """
        cpu_model = cpuinfo.get_cpu_info().get("brand_raw", "N/A")
        cpu_cores = psutil.cpu_count(logical=True)
        gpu_model, gpu_cores = "N/A", "N/A"
        if NVIDIA_SMI_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_model = pynvml.nvmlDeviceGetName(handle)
                    try:
                        gpu_cores = pynvml.nvmlDeviceGetMultiProcessorCount(handle)
                    except pynvml.NVMLError:
                        gpu_cores = "N/A (older card)"
            except pynvml.NVMLError:
                gpu_model = "NVIDIA driver issue"
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        return {
            "CPUmodel": cpu_model,
            "Number of CPU cores": cpu_cores,
            "GPU model": gpu_model,
            "Number of GPU cores": gpu_cores,
            "Memory available (GB)": memory_gb,
        }

    def _ensure_header(self):
        """
        Ensures the CSV file exists and has a header row.

        If the file does not exist, it is created and the headers defined in
        `self.headers` are written to it.
        """
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def track(self, start, end, title, verbose=False):
        """
        Records a single tracking entry to the CSV file.

        Args:
            start (float): The start time (from `time.perf_counter()`).
            end (float): The end time (from `time.perf_counter()`).
            title (str): A descriptive title for the tracked event.
            verbose (bool, optional): If True, prints a summary to the console.
                                      Defaults to False.
        """
        runtime_seconds = end - start
        runtime_formatted = time.strftime("%H:%M:%S", time.gmtime(runtime_seconds))
        data_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Runtime(format HH:MM:SS)": runtime_formatted,
            "Title": title,
            **self.system_specs,
        }
        with open(self.file_name, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(data_row)
        if verbose:
            self._print_verbose_summary(title, runtime_seconds, runtime_formatted)

    @contextlib.contextmanager
    def time(self, title, verbose=False):
        """
        A context manager to time a block of code.

        Example:
            tracker = GreenCodeTracker()
            with tracker.time("data_processing", verbose=True):
                # Code to be timed
                time.sleep(1)

        Args:
            title (str): A descriptive title for the timed block.
            verbose (bool, optional): If True, prints a summary to the console
                                      upon exiting the context. Defaults to False.
        """
        start_time = perf_counter()
        try:
            yield
        finally:
            end_time = perf_counter()
            self.track(start_time, end_time, title, verbose=verbose)

    def _print_verbose_summary(self, title, runtime_seconds, runtime_formatted):
        """
        Prints a formatted summary of the tracking results to the console.

        Args:
            title (str): The title of the tracked event.
            runtime_seconds (float): The total runtime in seconds.
            runtime_formatted (str): The runtime formatted as HH:MM:SS.
        """
        print("\n" + "=" * 150)
        print(
            f"                    GREEN CODE INITIATIVE - {title}                          "
        )
        print("-" * 150)
        print(
            f"Total time taken for {title}: {runtime_seconds:.3f} seconds ({runtime_formatted})"
        )
        print(
            f"CPU: {self.system_specs['CPUmodel']} ({self.system_specs['Number of CPU cores']} cores)"
        )
        if self.system_specs["GPU model"] != "N/A":
            print(
                f"GPU: {self.system_specs['GPU model']} ({self.system_specs['Number of GPU cores']} multiprocessors)"
            )
        print(f"Memory: {self.system_specs['Memory available (GB)']} GB")
        print(f"\n{title} complete. All results saved to {self.file_name}")
        print("=" * 150 + "\n")
