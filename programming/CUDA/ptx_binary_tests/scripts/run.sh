#! /bin/bash

# Run script in build directory
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/../build"

# Get params
source ../scripts/param.sh

# NVCC
echo "Running ${FILE}_nvcc..."
./${FILE}_nvcc

# Clang
if ! command -v clang++ &> /dev/null
then
    exit
fi
echo "Running ${FILE}_clang..."
./${FILE}_clang
