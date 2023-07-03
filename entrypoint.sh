#!/bin/bash
set -e

eval $( fixuid -q )
exec python -m baler "$@"
