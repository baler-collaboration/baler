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
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# System deps:
RUN pip install "poetry"

# Copy only requirements to cache them in docker layer
WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./

# Project initialization:
RUN poetry install --no-interaction --no-ansi

# Creating folders, and files for a project:
COPY ./baler /baler-root/baler

## -----------------------------------------------------------------------------
## Baler layer

FROM python:3.8-slim

# Copy Venv
COPY --from=python-base $VENV_PATH $VENV_PATH
COPY --from=python-base /baler-root/baler /baler-root/baler

ENTRYPOINT ["python", "baler"]
