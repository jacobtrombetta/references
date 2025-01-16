#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <utility>

#include "ptx.cuh"
#include "stats.h"

#define CUDA_CALLABLE __host__ __device__
using uint128_t = __uint128_t;

namespace sxt::bast {
//--------------------------------------------------------------------------------------------------
// narrow_cast
//--------------------------------------------------------------------------------------------------
template <class T, class U> constexpr T CUDA_CALLABLE narrow_cast(U&& u) noexcept {
  return static_cast<T>(std::forward<U>(u));
}
} // namespace sxt::bast

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// cmov
//--------------------------------------------------------------------------------------------------
/*
 Replace (f,g) with (g,g) if b == 1.
 Replace (f,g) with (f,g) if b == 0.
 *
 Preconditions: b in {0,1}.
 */
template <typename T>
CUDA_CALLABLE inline void cmov(T& f, const T g, unsigned int b) noexcept {
  const T mask = static_cast<T>(-static_cast<T>(b));
  f = f ^ (mask & (f ^ g));
}
} // namespace sxt::basn


namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// mac
//--------------------------------------------------------------------------------------------------
/**
 * @brief Multiply and carry.
*/
CUDA_CALLABLE void inline mac(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
  uint128_t ret_tmp = uint128_t{a} + (uint128_t{b} * uint128_t{c}) + uint128_t{carry};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
}

//--------------------------------------------------------------------------------------------------
// mul_wide_limbs_loops
//--------------------------------------------------------------------------------------------------
template <unsigned num_limbs>
static __device__ __forceinline__ void mul_wide_limbs_loops(uint64_t* c, const uint64_t* f, const uint64_t* g) noexcept {
# pragma unroll
  for (unsigned i = 0; i < num_limbs; ++i) {
    basfld::u64::addc_cc(0,0);
# pragma unroll
    for (unsigned j = 0; j < num_limbs; ++j) {
      c[i + j] = basfld::u64::madc_lo_cc(f[i], g[j], c[i + j]);
    }
    c[i + num_limbs] = basfld::u64::addc_cc(0,0);
    basn::cmov(c[i + num_limbs], uint64_t{0}, i == 0);
# pragma unroll
    for (unsigned j = 0; j < num_limbs; ++j) {
      c[i + j + 1] = basfld::u64::madc_hi_cc(f[i], g[j], c[i + j + 1]);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// mul_wide_limbs_unrolled
//--------------------------------------------------------------------------------------------------
template <unsigned num_limbs>
__device__ __forceinline__ void mul_wide_limbs_unrolled(uint64_t* c, const uint64_t* f, const uint64_t* g) noexcept {
  // i = 0
  basfld::u64::addc(0,0);
  c[0] = basfld::u64::mad_lo_cc(f[0], g[0], c[0]);
  c[1] = basfld::u64::madc_lo_cc(f[0], g[1], c[1]);
  c[2] = basfld::u64::madc_lo_cc(f[0], g[2], c[2]);
  c[3] = basfld::u64::madc_lo_cc(f[0], g[3], c[3]);
  basfld::u64::addc(0,0);
  c[1] = basfld::u64::mad_hi_cc(f[0], g[0], c[1]);
  c[2] = basfld::u64::madc_hi_cc(f[0], g[1], c[2]);
  c[3] = basfld::u64::madc_hi_cc(f[0], g[2], c[3]);
  c[4] = basfld::u64::madc_hi_cc(f[0], g[3], c[4]);

  // i = 1
  basfld::u64::addc(0,0);
  c[1] = basfld::u64::mad_lo_cc(f[1], g[0], c[1]);
  c[2] = basfld::u64::madc_lo_cc(f[1], g[1], c[2]);
  c[3] = basfld::u64::madc_lo_cc(f[1], g[2], c[3]);
  c[4] = basfld::u64::madc_lo_cc(f[1], g[3], c[4]);
  c[5] = basfld::u64::addc(0,0);
  c[2] = basfld::u64::mad_hi_cc(f[1], g[0], c[2]);
  c[3] = basfld::u64::madc_hi_cc(f[1], g[1], c[3]);
  c[4] = basfld::u64::madc_hi_cc(f[1], g[2], c[4]);
  c[5] = basfld::u64::madc_hi_cc(f[1], g[3], c[5]);

  // i = 2
  basfld::u64::addc(0,0);
  c[2] = basfld::u64::mad_lo_cc(f[2], g[0], c[2]);
  c[3] = basfld::u64::madc_lo_cc(f[2], g[1], c[3]);
  c[4] = basfld::u64::madc_lo_cc(f[2], g[2], c[4]);
  c[5] = basfld::u64::madc_lo_cc(f[2], g[3], c[5]);
  c[6] = basfld::u64::addc(0,0);
  c[3] = basfld::u64::mad_hi_cc(f[2], g[0], c[3]);
  c[4] = basfld::u64::madc_hi_cc(f[2], g[1], c[4]);
  c[5] = basfld::u64::madc_hi_cc(f[2], g[2], c[5]);
  c[6] = basfld::u64::madc_hi_cc(f[2], g[3], c[6]);

  // i = 3
  basfld::u64::addc(0,0);
  c[3] = basfld::u64::mad_lo_cc(f[3], g[0], c[3]);
  c[4] = basfld::u64::madc_lo_cc(f[3], g[1], c[4]);
  c[5] = basfld::u64::madc_lo_cc(f[3], g[2], c[5]);
  c[6] = basfld::u64::madc_lo_cc(f[3], g[3], c[6]);
  c[7] = basfld::u64::addc(0,0);
  c[4] = basfld::u64::mad_hi_cc(f[3], g[0], c[4]);
  c[5] = basfld::u64::madc_hi_cc(f[3], g[1], c[5]);
  c[6] = basfld::u64::madc_hi_cc(f[3], g[2], c[6]);
  c[7] = basfld::u64::madc_hi_cc(f[3], g[3], c[7]);

  // for (unsigned k = 0; k < 7; ++k) {
  //     printf("c[%d] = 0x%lx\n", k, c[k]);
  // }
}

//--------------------------------------------------------------------------------------------------
// mul_wide_4_limbs_unrolled
//--------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mul_wide_4_limbs_unrolled(uint64_t* t, const uint64_t* f, const uint64_t* g) noexcept {
  uint64_t carry{0};

  basfld::mac(t[0], carry, 0, f[0], g[0]);
  basfld::mac(t[1], carry, 0, f[0], g[1]);
  basfld::mac(t[2], carry, 0, f[0], g[2]);
  basfld::mac(t[3], carry, 0, f[0], g[3]);
  t[4] = carry;
  
  carry = 0;
  basfld::mac(t[1], carry, t[1], f[1], g[0]);
  basfld::mac(t[2], carry, t[2], f[1], g[1]);
  basfld::mac(t[3], carry, t[3], f[1], g[2]);
  basfld::mac(t[4], carry, t[4], f[1], g[3]);
  t[5] = carry;
  
  carry = 0;
  basfld::mac(t[2], carry, t[2], f[2], g[0]);
  basfld::mac(t[3], carry, t[3], f[2], g[1]);
  basfld::mac(t[4], carry, t[4], f[2], g[2]);
  basfld::mac(t[5], carry, t[5], f[2], g[3]);
  t[6] = carry;
  
  carry = 0;
  basfld::mac(t[3], carry, t[3], f[3], g[0]);
  basfld::mac(t[4], carry, t[4], f[3], g[1]);
  basfld::mac(t[5], carry, t[5], f[3], g[2]);
  basfld::mac(t[6], carry, t[6], f[3], g[3]);
  t[7] = carry;
}
} // namespace sxt::basfld

//--------------------------------------------------------------------------------------------------
// sxt_mul_wide_limbs_loops
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mul_wide_limbs_loops(uint64_t* __restrict__ ret,
                                       const uint64_t* __restrict__ a,
                                       const uint64_t* __restrict__ b) {
  sxt::basfld::mul_wide_limbs_loops<4>(ret, a, b);
}

//--------------------------------------------------------------------------------------------------
// sxt_mul_wide_limbs_unrolled
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mul_wide_limbs_unrolled(uint64_t* __restrict__ ret,
                                            const uint64_t* __restrict__ a,
                                            const uint64_t* __restrict__ b) {
  sxt::basfld::mul_wide_limbs_unrolled<4>(ret, a, b);
}

//--------------------------------------------------------------------------------------------------
// sxt_mul_wide_4_limbs_unrolled
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mul_wide_4_limbs_unrolled(uint64_t* __restrict__ ret,
                                              const uint64_t* __restrict__ a,
                                              const uint64_t* __restrict__ b) {
  sxt::basfld::mul_wide_4_limbs_unrolled(ret, a, b);
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main() {
  uint64_t* a;
  uint64_t* b;
  uint64_t* c;
  uint64_t* ret_wide_limbs_old;
  uint64_t* ret_wide_limbs;
  uint64_t* ret_wide_limbs_working;

  constexpr unsigned num_limbs = 4;
  cudaMallocManaged(&a, sizeof(uint64_t) * num_limbs);
  cudaMallocManaged(&b, sizeof(uint64_t) * num_limbs);
  cudaMallocManaged(&c, sizeof(uint64_t) * num_limbs);
  cudaMallocManaged(&ret_wide_limbs_old, sizeof(uint64_t) * 2 * num_limbs);
  cudaMallocManaged(&ret_wide_limbs, sizeof(uint64_t) * 2 * num_limbs);
  cudaMallocManaged(&ret_wide_limbs_working, sizeof(uint64_t) * 2 * num_limbs);

  // Create random data
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  for (unsigned i = 0; i < num_limbs; ++i) {
    a[i] = dis(gen);
    b[i] = dis(gen);
    c[i] = dis(gen);
  }

  sxt_mul_wide_limbs_loops<<<1,1>>>(ret_wide_limbs_old, a, b);
  sxt_mul_wide_limbs_unrolled<<<1,1>>>(ret_wide_limbs, a, b);
  sxt_mul_wide_4_limbs_unrolled<<<1,1>>>(ret_wide_limbs_working, a, b);
  cudaDeviceSynchronize();

  // Print data
  std::cout << "hex(0x";
  for (unsigned j = num_limbs; j > 0; --j) {
    std::cout << std::hex << a[j-1];
  }
  std::cout << " * 0x";

  for (unsigned j = num_limbs; j > 0; --j) {
    std::cout << std::hex << b[j-1];
  }
  std::cout << ")" << std::endl;

  std::cout << "0x";
  for (unsigned j = 2*num_limbs; j > 0; --j){
    std::cout << std::hex << " " << ret_wide_limbs_old[j-1];
  }
  std::cout << " = ret_wide_limbs_loops" << std::endl;

  std::cout << "0x";
  for (unsigned j = 2*num_limbs; j > 0; --j){
    std::cout << std::hex << " " << ret_wide_limbs[j-1];
  }
  std::cout << " = ret_wide_limbs_unrolled" << std::endl;

  std::cout << "0x";
  for (unsigned j = 2*num_limbs; j > 0; --j){
    std::cout << std::hex << " " << ret_wide_limbs_working[j-1];
  }
  std::cout << " = ret_wide_limbs_unrolled_mac" << std::endl;

  return 0;
}
