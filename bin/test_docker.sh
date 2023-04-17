#!/bin/bash

# This script servers as a wrapper around docker run to run the docker image
# It also tests the docker image

WORKSPACE_NAME="CMS_example"
PROJECT_NAME="example_CMS"
DOCKERFILE="Dockerfile"
ARCH="amd64"

# Check if we're on ARM
if [ "$(uname -m)" = "arm64" ]; then
  ARCH="arm64"
fi

# TODO Fix this
# Run the docker image with --mode=train and --project=example_CMS
# We need to mount the projects directory and the data directory
docker run -v $(pwd)/projects:/baler-root/projects \
           baler-${ARCH}:latest --mode train --project "${WORKSPACE_NAME}" "${PROJECT_NAME}"

docker run -v $(pwd)/projects:/baler-root/projects \
           baler-${ARCH}:latest --mode compress --project "${WORKSPACE_NAME}" "${PROJECT_NAME}"

docker run -v $(pwd)/projects:/baler-root/projects \
           baler-${ARCH}:latest --mode decompress --project "${WORKSPACE_NAME}" "${PROJECT_NAME}"
