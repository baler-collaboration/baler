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
## Builder: resolve the locked environment with uv

# The multi-stage layout was originally based on
# https://github.com/michaeloliverx/python-poetry-docker-example/blob/master/docker/Dockerfile
# and has since been ported from poetry to uv.

FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/baler-root/baler/.venv

WORKDIR /baler-root/baler

# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY baler/ ./baler/
COPY README.md LICENSE NOTICE ./
RUN uv sync --frozen --no-dev --no-editable

## -----------------------------------------------------------------------------
## Baler layer

FROM python:3.10-slim

# Copy the virtual environment (baler is installed into it as a regular package)
COPY --from=builder /baler-root/baler/.venv /baler-root/baler/.venv

# Configure run time
ENV PYTHONUNBUFFERED=1 \
    PATH="/baler-root/baler/.venv/bin:$PATH"
WORKDIR /baler-root/

# Install fixuid
COPY fixuid.sh fixuid.sh
RUN ["./fixuid.sh"]

COPY entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
