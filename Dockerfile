# Based on https://github.com/michaeloliverx/python-poetry-docker-example/blob/master/docker/Dockerfile

## -----------------------------------------------------------------------------
## Base image with VENV

FROM python:3.8-slim as python-base

# Configure environment

ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/baler-root/baler" \
    VENV_PATH="/baler-root/baler/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# System deps:
RUN pip install "poetry"

# Copy only requirements to cache them in docker layer
WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./

# Project initialization:
RUN poetry install --no-interaction --no-ansi

# Creating folders, and files for the project:
COPY ./baler/ __init__.py README.md ./tests/ ./

# Creating python wheel
RUN poetry build

## -----------------------------------------------------------------------------
## Baler layer

FROM python:3.8-slim

# Copy virtual environment
WORKDIR /baler-root/baler
COPY --from=python-base /baler-root/baler/dist/*.whl ./

# Install wheel
RUN pip install *.whl

# Copy source 
COPY --from=python-base /baler-root/baler/modules/ ./modules
COPY --from=python-base /baler-root/baler/*.py /baler-root/baler/README.md ./

# Configure run time
ENV PYTHONUNBUFFERED=1
WORKDIR /baler-root/

# Configure fixuid env
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 \
    --ingroup docker \
    --home /home/docker \
    --shell /bin/sh \
    --disabled-password \
    --gecos "" \
    docker

# Install fixuid
RUN apt update && \
    apt install --no-install-recommends -y curl && \
    USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml

USER docker:docker
ENTRYPOINT ["sh", "-c", "fixuid && python baler"]
