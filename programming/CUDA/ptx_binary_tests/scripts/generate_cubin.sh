#! /bin/bash

# Run script in build directory
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/../build"

# Get params
source ../scripts/param.sh

# NVCC
echo "Building NVCC CUBIN for ${FILE}.cu"
nvcc -O3 -cubin -arch=$ARCH ../$FILE.cu -o ${FILE}_nvcc.cubin
cuobjdump -sass ${FILE}_nvcc.cubin > ${FILE}_nvcc_cubin.txt

# Clang
if ! command -v clang++ &> /dev/null
then
    exit
fi
echo "Building Clang CUBIN for ${FILE}.cu"
clang++ -O3 -c ../$FILE.cu --cuda-gpu-arch=$ARCH -o ${FILE}_clang.cubin
cuobjdump -sass ${FILE}_clang.cubin > ${FILE}_clang_cubin.txt
