# Baler on Windows Native+CUDA (GPU) - Experimental Guide

This documentation provides a step-by-step guide on running Baler with Windows Native+CUDA (GPU). Please follow the instructions carefully to ensure successful setup and execution.

## Prerequisites:

- Windows OS (optional: with CUDA compatible GPU installed).
- Ensure you have Git installed on your system for cloning the Baler project.

## Setup:

### STEP 1: Install Python3.10

Download and install Python 3.10 from the official website using the following link:

[Python 3.10.11](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)

### STEP 2: Install Poetry

Once Python is installed, open your terminal or command prompt and run the following command to install Poetry:

```console
python -m pip install poetry
```

### STEP 3: Clone Baler Project

Refer to the primary README documentation of the Baler project for detailed instructions on cloning the repository using Git.

### STEP 4: Prepare Baler Project for GPU

Navigate to the cloned Baler directory. If you find a `poetry.lock` file in the directory, delete it.

Now, open the `pyproject.toml` file using your preferred text editor.

### STEP 5: (Optional) Update PyTorch Version for CUDA

In the `pyproject.toml` file, find the torch dependency version and update it with the following URL:

Replace this:
```
torch = ">=2.0.0, !=2.0.1"
```

With this:
```
torch = { url = "https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp310-cp310-win_amd64.whl#sha256=5ee2b7c19265b9c869525c378fcdf350510b8f3fc08af26da1a2587a34cea8f5"}
```

> **Note**: This step is particularly important if you want to run your training on GPUs. Please ensure that the version you are using (in this case: PyTorch 2.0.0 with CUDA 11.8 (cu118) for Python 3.10 (cp310)) corresponds to the version you want to run.

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

### STEP 6: Install Project Dependencies

Inside the Baler directory, execute the following command to install the required dependencies:

```console
python -m poetry install
```

### STEP 7: Run Baler with Poetry

Once all dependencies are installed, you can run Baler with the following command:

```console
python -m poetry run baler baler [-h] --mode MODE --project WORKSPACE PROJECT [--verbose]
```

Replace `MODE`, `WORKSPACE`, and `PROJECT` with appropriate values as per your requirement.

---

Thank you for using Baler! If you face any issues, please refer to the project's GitHub issues section or contact the project maintainers.
