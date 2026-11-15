## Python Setup
A setup for windows is available [here](documentation/setup/python_setup_windows.md)

For some Linux users, disable the KDE keyring
```console
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing the python environment. uv installs a suitable Python version (3.10) itself if needed.

2. Clone this repository and move into the project
```console
git clone https://github.com/baler-collaboration/baler.git && cd baler
```

3. Install dependencies using uv (this creates a `.venv/` in the project, following `uv.lock`)
```console
uv sync
```

4. Try the installation with `uv run baler`, this should give the following output:

```console
usage: baler [-h] --mode MODE --project WORKSPACE PROJECT [--verbose]
baler: error: the following arguments are required: --mode, --project
```

## Working Example with Python

Here we provide some instructions for our working examples.

### Computational Fluid Dynamics Example

#### Training ####
To train the autoencoder to compress your data, you run the following command. The config file `./workspaces/CFD_workspace/CFD_project_v1/config/CFD_project_v1_config.py`. This details the path of the data, the number of epochs, and all the other training parameters.
```console
uv run baler --project CFD_workspace CFD_project_animation --mode train
```

#### Compressing ####
To use the derived model for compression, you can now choose ``--mode compress``, which can be run as

```console
uv run baler --project CFD_workspace CFD_project_animation --mode compress
```
This will output a compressed file called "compressed.pickle", and this is the latent space representation of the input dataset. It will also output cleandata_pre_comp.pickle which is just the exact data being compressed.

#### Decompressing ####
To decompress the compressed file, we choose --mode decompress and run:
```console
uv run baler --project CFD_workspace CFD_project_animation --mode decompress
```

#### Plotting ####
To plot the difference of your variables before compression and after decompression, we can use the following command to generate a .pdf document under ``./workspaces/firstWorkspace/firstProject/output/plotting/comparison.pdf``

```console
uv run baler --project CFD_workspace CFD_project_animation --mode plot
```

### High Energy Physics Example ###
To run our High Energy Physics using CMS data (DOI:10.7483/OPENDATA.CMS.KL8H.HFVH) follow the above instructions but replace `--project CFD_workspace CFD_project_animation` with `--project CMS_workspace CMS_project_v1`

## New project ##
To create the folder structure and a skeleton config for your own project:
```console
uv run baler --project firstWorkspace firstProject --mode newProject
```

## Optional: FPGA conversion with hls4ml ##
`--mode convert_with_hls4ml` converts a trained model to an FPGA (HLS) project with [hls4ml](https://github.com/fastmachinelearning/hls4ml). It needs extra dependencies, available on **Python 3.10-3.13 only**:
```console
uv sync --extra hls4ml
```
Synthesis also needs Vivado 2020.1, and the project config needs the options listed in the docstring of `perform_hls4ml_conversion` (`baler/modules/helper.py`).

> **Note:** this integration has not been worked on since mid-2023. It was last checked in November 2026 (model conversion and HLS project writing with hls4ml 0.7.1, without Vivado); synthesis was not tested.

## Running the tests ##
```console
uv run pytest                         # unit tests
uv run tests/roundtrip/run_roundtrip.sh   # full train/compress/decompress/plot round trip on the CMS example
```

CI (`.github/workflows/test_and_lint.yaml`) runs black, pytest on the lowest and highest supported Python, and the round trip on the newest Python. The round-trip thresholds in `tests/roundtrip/thresholds.json` are calibrated from several CI runs. To recalibrate (e.g. after dependency updates), start the workflow manually with `roundtrip_runs` set to `[1,2,3,4,5]`, download the `roundtrip-results-run*` artifacts, and run:
```console
uv run tests/roundtrip/calibrate_thresholds.py <dir>/roundtrip-results-run*/roundtrip_results/roundtrip_metrics.json \
    --out tests/roundtrip/thresholds.json --source "<link to the workflow run>"
```
