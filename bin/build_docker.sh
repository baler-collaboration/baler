#!/bin/bash

set -e

# This script serves as a wrapper around docker build to build the docker image
# And is run in the GitHub Actions workflow

DOCKERFILE="${DOCKERFILE:-Dockerfile}"
ARCH="${ARCH:-$(uname -m)}"
TAG="baler-${ARCH}:latest"

if ! [ -x "$(command -v docker)" ]; then
  echo 'Error: docker is not installed.' >&2
  exit 1
fi

if [ "${ARCH}" = "arm64" ]; then
  DOCKERFILE="${DOCKERFILE}.${ARCH}"
fi

docker build -t ${TAG} -f "${DOCKERFILE}" .
echo "Successfully built ${TAG}..."
