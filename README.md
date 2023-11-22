[![DOI](https://zenodo.org/badge/576188110.svg)](https://zenodo.org/badge/latestdoi/576188110)\
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)\
![example workflow](https://github.com/baler-compressor/baler/actions/workflows/test_and_lint.yaml/badge.svg)
![example workflow](https://github.com/baler-compressor/baler/actions/workflows/docker.yaml/badge.svg)

# Introduction
Baler is a tool used to test the feasibility of compressing different types of scientific data using machine learning-based autoencoders. Baler provides you with an easy way to:
1. Train a machine learning model on your data
2. Compress your data with that model. This will also save the compressed file and model
3. Decompress the file using the model at a later time
4. Plot the performance of the compression/decompression


# Getting Started #
**NOTE:** For the same performance and version as presented in our [Arxiv](https://arxiv.org/abs/2305.02283) paper, please use release [v1.0.0](https://github.com/baler-collaboration/baler/tree/v1.0.0) and the setup instructions given there. v1.0.0 also has a working docker implementation. We are currently experiencing some performance issues on the main branch compared.

In the links below we offer instructions on how to set up Baler and working tutorial examples to get you started. We offer two ways to run baler:
* [Python](docs/setup/python_setup.md)
* [Docker/Singularity/Apptainer](docs/setup/docker_setup.md)



# Contributing

If you wish to contribute, please see the [contribution guidelines](https://github.com/baler-collaboration/baler/blob/main/docs/CONTRIBUTING.md).
