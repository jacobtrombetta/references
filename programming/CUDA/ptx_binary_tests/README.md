# PTX and Binary tests

## Full limb
Clang write a lot of values to global when unrolling. `mac` with NVCC performs the best.

## Multiply and carry
Tests from `mac_ptx.cu`. Goal is to identify PTX and binary code differences between different `mac` implementations.

- PTX implementations use less registers because of reuse of `lo` and `hi`.
- `mac` has as many instructions as `mac_ptx_2` returning a value (no references)
- All functions produce the same binary code on Ampere architecture (GeForce RTX 3080).

All data in `results/mac_ptx` folder by architecture.

### Conclusion
PTX does not help with the `mac` function using either the Clang or NVCC compiler.

### To run full suite of tests
1. Update `scripts/param.sh` to point to `FILE=mac_ptx`.
1. Run Docker container: `./ci/run_docker_gpu.sh`
1. Execute build: `./scripts/build.sh`
1. Run tests: `./scripts/run.sh`
1. Build PTX: `./scripts/generate_ptx.sh`
1. Build cubin: `./scripts/generate_cubin.sh`
