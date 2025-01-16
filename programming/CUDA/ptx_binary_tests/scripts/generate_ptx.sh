#! /bin/bash

# Run script in build directory
SCRIPT_DIR="$(dirname "$0")"
cd  "$SCRIPT_DIR/../build"

# Get params
source ../scripts/param.sh

# NVCC
echo "Building NVCC PTX for ${FILE}.cu"
nvcc -O3 -ptx -arch=$ARCH ../$FILE.cu -o ${FILE}_nvcc.ptx

# Clang
if ! command -v clang++ &> /dev/null
then
    exit
fi
echo "Building Clang PTX for ${FILE}.cu"
clang++ -O3 -emit-llvm -c -S ../$FILE.cu --cuda-gpu-arch=$ARCH
llc -mcpu=$ARCH ${FILE}-cuda-nvptx64-nvidia-cuda-${ARCH}.bc -o ${FILE}_clang.ptx
rm ${FILE}-cuda-nvptx64-nvidia-cuda-${ARCH}.bc
rm $FILE.ll
