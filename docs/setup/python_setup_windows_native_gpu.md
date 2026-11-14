# Baler on Windows Native+CUDA (GPU) - Experimental Guide

This documentation provides a step-by-step guide on running Baler with Windows Native+CUDA (GPU). Please follow the instructions carefully to ensure successful setup and execution.

## Prerequisites:

- Windows OS (optional: with CUDA compatible GPU installed).
- Ensure you have Git installed on your system for cloning the Baler project.

## Setup:

### STEP 1: Install Python3.10

Download and install Python 3.10 from the official website using the following link:

[Python 3.10.11](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)

### STEP 2: Install uv

Once Python is installed, open your terminal or command prompt and run the following command to install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```console
python -m pip install uv
```

### STEP 3: Clone Baler Project

Refer to the primary README documentation of the Baler project for detailed instructions on cloning the repository using Git.

### STEP 4: Install Project Dependencies

Navigate to the cloned Baler directory and run:

```console
uv sync
```

This creates a virtual environment in `.venv\` with the exact versions pinned in `uv.lock`. Do not delete or edit `uv.lock` or `pyproject.toml`.

### STEP 5: (Optional) Install the CUDA build of PyTorch

The torch wheel installed in STEP 4 is CPU-only on Windows. To train on an NVIDIA GPU, replace it with the CUDA build of the **same torch version**:

```console
uv run python -c "import torch; print(torch.__version__)"
uv pip install --reinstall-package torch "torch==<VERSION>" --index-url https://download.pytorch.org/whl/cu124
```

Replace `<VERSION>` with the version printed by the first command (e.g. `2.6.0`). Pick the CUDA variant (`cu118`, `cu124`, `cu126`, ...) matching your driver from the [PyTorch install page](https://pytorch.org/get-started/locally/). Check it worked with:

```console
uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"
```

> **Note**: plain `uv run` and `uv sync` reset the environment to `uv.lock` and therefore put the CPU-only torch back. After STEP 5, always use `uv run --no-sync` (as below), or activate the environment with `.venv\Scripts\activate` and call `baler` directly.

### STEP 5a: (Optional) Disable Energy Profiling

If you wish to avoid profiling the energy usage (especially when encountering memory issues with GPU acceleration), open the `baler/baler.py` file and comment out lines 79 and 80. These lines should look like:

```python
@pytorch_profile
@energy_profiling(project_name="baler_training", measure_power_secs=1)
```

To comment them, simply place a `#` in front of each line:

```python
# @pytorch_profile
# @energy_profiling(project_name="baler_training", measure_power_secs=1)
```

> **Note**: This step is particularly important if you face out-of-memory issues when utilizing GPU acceleration or encounter memory-related problems in general.

### STEP 6: Run Baler

Once all dependencies are installed, you can run Baler with the following command (drop `--no-sync` if you skipped STEP 5):

```console
uv run --no-sync baler --mode MODE --project WORKSPACE PROJECT [--verbose]
```

Replace `MODE`, `WORKSPACE`, and `PROJECT` with appropriate values as per your requirement.

---

Thank you for using Baler! If you face any issues, please refer to the project's GitHub issues section or contact the project maintainers.
