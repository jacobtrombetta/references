#! /bin/bash

# Run script in build directory
SCRIPT_DIR="$(dirname "$0")"
mkdir -p  "$SCRIPT_DIR/../build"
cd  "$SCRIPT_DIR/../build"

# Get ARCH from param.sh
source ../scripts/param.sh

# NVCC
echo "Building with NVCC version:"
nvcc --version
echo ""
echo "...building ${FILE}..."
nvcc -O3 -arch=$ARCH ../$FILE.cu -o ${FILE}_nvcc

# Clang
if ! command -v clang++ &> /dev/null
then
    exit
fi
echo ""
echo "Building with Clang version:"
clang++ --version
echo ""
echo "...building ${FILE}..."
clang++ -O3 --cuda-gpu-arch=$ARCH -I/usr/local/cuda/include \
        -L/usr/local/cuda/lib64 -lcudart ../$FILE.cu -o ${FILE}_clang
