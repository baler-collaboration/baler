FROM python:3.8-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on 

# System deps:
RUN pip install "poetry"

# Copy only requirements to cache them in docker layer
WORKDIR /baler-root
COPY poetry.lock pyproject.toml /baler-root/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Creating folders, and files for a project:
COPY ./baler /baler-root/baler

ENTRYPOINT ["python", "baler"]
