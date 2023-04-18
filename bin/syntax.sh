#!/bin/bash

# This script is used to run the code quality tools on the codebase.

# Variables
CODEBASE="baler"
BLACK_OPTIONS="--check"
BLACK_APPLY_OPTIONS=""

# Function to check if the script is running in a CI environment
function is_ci() {
  if [ -n "$CI" ]; then
    return 0
  else
    return 1
  fi
}

# Function to run Black code formatter
function run_black() {
  echo "Running Black on the codebase..."
  if poetry run black ${BLACK_OPTIONS} ${CODEBASE}; then
    echo "Black completed successfully."
  else
    if is_ci; then
      echo "Black found issues in the codebase. Please fix them before committing."
      exit 1
    else
      local apply_fix
      echo "Black found issues in the codebase."
      read -p "Would you like to apply Black to fix the issues? (Y/n) " apply_fix

      if [[ "$apply_fix" =~ ^([yY][eE][sS]|[yY])$ ]] || [[ -z "$apply_fix" ]]; then
        echo "Applying Black to the codebase..."
        poetry run black ${BLACK_APPLY_OPTIONS} ${CODEBASE}
        echo "Black successfully applied to the codebase."
      else
        echo "Skipping Black auto-fix. Please fix the issues manually before committing."
      fi
    fi
  fi
}

# Main script
function main() {
  run_black
}

# Run the main function
main
