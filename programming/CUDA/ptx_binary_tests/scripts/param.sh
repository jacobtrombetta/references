# File to compile
FILE=full_limb_mul
export FILE

# Architecture for the target GPU
# Compute capability for the target GPU: https://developer.nvidia.com/cuda-gpus
# GeForce RTX 3080 - Ampere, Compute Capability 8.6, sm_86: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/
# A1000 - Ampere, Compute Capability 8.0, sm_80: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
# Ampere instruction set: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#turing-instruction-set
# Tesla T4 - Turing, Compute Capability 7.5, sm_75: https://www.nvidia.com/en-us/data-center/tesla-t4/
# Turning instruction set: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#turing-instruction-set
# Blitzar uses - sm_70
ARCH=sm_70
export ARCH
