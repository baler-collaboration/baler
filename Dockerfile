# MIT License

# Copyright (c) 2020 Michael Oliver

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## -----------------------------------------------------------------------------
# The following code was based on
# https://github.com/michaeloliverx/python-poetry-docker-example/blob/master/docker/Dockerfile
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

COPY entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
