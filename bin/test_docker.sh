#!/bin/bash

# This script serves as a wrapper around docker run to run the docker image
# It also tests the docker image

DOCKERFILE="${DOCKERFILE:-Dockerfile}"
ARCH="${ARCH:-$(uname -m)}"
WORKSPACE_NAME="${WORKSPACE_NAME:-CMS_example}"
PROJECT_NAME="${PROJECT_NAME:-example_CMS}"
DOCKER_IMAGE="baler-${ARCH}:latest"

function run_docker() {
  local mode="$1"
  docker run -v "$(pwd)/workspaces:/baler-root/workspaces" \
             ${DOCKER_IMAGE} --mode "${mode}" --project "${WORKSPACE_NAME}" "${PROJECT_NAME}"
}

echo "Training the project..."
run_docker "train"

echo "Compressing the project..."
run_docker "compress"

echo "Decompressing the project..."
run_docker "decompress"

echo "All tasks completed successfully."
