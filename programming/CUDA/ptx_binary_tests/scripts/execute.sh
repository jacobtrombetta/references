# Runs everything

#! /bin/bash

# Run script in scripts directory
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

./build.sh
./generate_ptx.sh
./generate_cubin.sh
./run.sh
