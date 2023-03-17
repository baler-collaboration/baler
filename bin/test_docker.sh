DOCKERFILE="Dockerfile"
ARCH="amd64"

# Check if we're on ARM
if [ "$(uname -m)" = "arm64" ]; then
  ARCH="arm64"
fi

# Run the docker image with --mode=train and --project=example_CMS
# We need to mount the projects directory and the data directory
docker run -v $(pwd)/projects:/baler-root/projects \
           -v $(pwd)/data:/baler-baler/data \
           baler-${ARCH}:latest --mode=train --project=example_CMS
