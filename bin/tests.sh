#!/bin/bash

# This script is used to run the tests.

# Exit the script if any command fails
set -e

# Variables
PYTEST_OPTIONS="--verbose"

# Function to run pytest
function run_pytest() {
  echo "Running pytest with options: ${PYTEST_OPTIONS}"
  poetry run pytest ${PYTEST_OPTIONS}
  echo "Pytest completed successfully."
}

# Main script
function main() {
  run_pytest
}

# Run the main function
main
