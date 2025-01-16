#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "ptx.cuh"

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
// mac_ptx_1
//--------------------------------------------------------------------------------------------------
/**
 * @brief Multiply and carry using 6 calls to single line PTX intrinsics, not mad calls.
*/
CUDA_CALLABLE void inline mac_ptx_1(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                                              const uint64_t c) noexcept {
#ifdef __CUDA_ARCH__
  asm volatile("{\n\t"                              // scope registers
                ".reg .u64 lo, hi;\n\t"              // create registers lo and hi
                "mul.lo.u64 lo, %2, %3;\n\t"         // lo = (b*c).lo
                "mul.hi.u64 hi, %2, %3;\n\t"         // hi = (b*c).hi
                "add.cc.u64 lo, lo, %4;\n\t"         // lo = lo + a -> CC.CF
                "addc.u64 hi, hi, 0;\n\t"            // hi = hi + CC.CF
                "add.cc.u64 lo, lo, %5;\n\t"         // lo = lo + carry -> CC.CF
                "addc.u64 hi, hi, 0;\n\t"            // hi = hi + CC.CF
                "mov.u64 %0, lo;\n\t"                // ret = lo
                "mov.u64 %1, hi;\n\t"                // carry = hi
                "}"                                  // end scope
                : "=l"(ret), "=l"(carry)             // outputs
                : "l"(b), "l"(c), "l"(a), "l"(carry) // inputs
  );
#else
  mac(ret, carry, a, b, c);
#endif
}

//--------------------------------------------------------------------------------------------------
// mac_ptx_2
//--------------------------------------------------------------------------------------------------
/**
 * @brief Multiply and carry no references.
*/
struct wide_limbs {
  uint64_t lo;
  uint64_t hi;
};
CUDA_CALLABLE wide_limbs inline mac_ptx_2(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t carry) noexcept {
  wide_limbs ret;
#ifdef __CUDA_ARCH__
  asm volatile("mul.lo.u64 %0, %2, %3;\n\t"         // lo = (b*c).lo
               "mul.hi.u64 %1, %2, %3;\n\t"         // hi = (b*c).hi
               "add.cc.u64 %0, %0, %4;\n\t"         // lo = lo + a -> CC.CF
               "addc.u64 %1, %1, 0;\n\t"            // hi = hi + CC.CF
               "add.cc.u64 %0, %0, %5;\n\t"         // lo = lo + carry -> CC.CF
               "addc.u64 %1, %1, 0;\n\t"            // hi = hi + CC.CF
               : "=l"(ret.lo), "=l"(ret.hi)         // outputs
               : "l"(b), "l"(c), "l"(a), "l"(carry) // inputs
  );
#else
  mac(ret.lo, ret.hi, a, b, c);
#endif
  return ret;
}

//--------------------------------------------------------------------------------------------------
// mac_ptx_3
//--------------------------------------------------------------------------------------------------
/**
 * @brief Multiply and carry using pointers
*/
CUDA_CALLABLE void inline mac_ptx_3(uint64_t* ret, uint64_t* carry, const uint64_t a, const uint64_t b,
                                    const uint64_t c) noexcept {
#ifdef __CUDA_ARCH__
  uint64_t ret_t = 0;
  uint64_t carry_t = *carry;
  asm volatile("{\n\t"                                // scope registers
               ".reg .u64 lo, hi;\n\t"                // create registers lo and hi
               "mul.lo.u64 lo, %2, %3;\n\t"           // lo = (b*c).lo
               "mul.hi.u64 hi, %2, %3;\n\t"           // hi = (b*c).hi
               "add.cc.u64 lo, lo, %4;\n\t"           // lo = lo + a -> CC.CF
               "addc.u64 hi, hi, 0;\n\t"              // hi = hi + CC.CF
               "add.cc.u64 lo, lo, %5;\n\t"           // lo = lo + carry -> CC.CF
               "addc.u64 hi, hi, 0;\n\t"              // hi = hi + CC.CF
               "mov.u64 %0, lo;\n\t"                  // ret = lo
               "mov.u64 %1, hi;\n\t"                  // carry = hi
               "}"                                    // end scope
               : "=l"(ret_t), "=l"(carry_t)           // outputs
               : "l"(b), "l"(c), "l"(a), "l"(carry_t) // inputs
  );
  *ret = ret_t;
  *carry = carry_t;
#else
  wide_limbs result;
  mac(result.lo, result.hi, a, b, c);
  *ret = result.lo;
  *carry = result.hi;
#endif  
}
} // namespace sxt::basfld

//--------------------------------------------------------------------------------------------------
// sxt_mac
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mac(uint64_t* __restrict__ ret,
                        uint64_t* __restrict__ carry, 
                        const uint64_t* __restrict__ a,
                        const uint64_t* __restrict__ b,
                        const uint64_t* __restrict__ c) {
    sxt::basfld::mac(ret[0], carry[0], a[0], b[0], c[0]);
}

//--------------------------------------------------------------------------------------------------
// sxt_mac_ptx_1
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mac_ptx_1(uint64_t* __restrict__ ret,
                              uint64_t* __restrict__ carry, 
                              const uint64_t* __restrict__ a,
                              const uint64_t* __restrict__ b,
                              const uint64_t* __restrict__ c) {
  sxt::basfld::mac_ptx_1(ret[0], carry[0], a[0], b[0], c[0]);
}

//--------------------------------------------------------------------------------------------------
// sxt_mac_ptx_2
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mac_ptx_2(uint64_t* __restrict__ ret,
                              uint64_t* __restrict__ carry, 
                              const uint64_t* __restrict__ a,
                              const uint64_t* __restrict__ b,
                              const uint64_t* __restrict__ c) {
  sxt::basfld::wide_limbs result = sxt::basfld::mac_ptx_2(a[0], b[0], c[0], carry[0]);
  ret[0] = result.lo;
  carry[0] = result.hi;
}

//--------------------------------------------------------------------------------------------------
// sxt_mac_ptx_3
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mac_ptx_3(uint64_t* __restrict__ ret,
                              uint64_t* __restrict__ carry, 
                              const uint64_t* __restrict__ a,
                              const uint64_t* __restrict__ b,
                              const uint64_t* __restrict__ c) {
  sxt::basfld::mac_ptx_3(&ret[0], &carry[0], a[0], b[0], c[0]);
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main() {
  // Create data
  uint64_t* a;
  uint64_t* b;
  uint64_t* c;
  uint64_t* car_orig;
  uint64_t* ret_orig;
  uint64_t* car_ptx_1;
  uint64_t* ret_ptx_1;
  uint64_t* car_ptx_2;
  uint64_t* ret_ptx_2;
  uint64_t* car_ptx_3;
  uint64_t* ret_ptx_3;

  // Copy data to device 
  cudaMallocManaged(&a, sizeof(uint64_t));
  cudaMallocManaged(&b, sizeof(uint64_t));
  cudaMallocManaged(&c, sizeof(uint64_t));
  cudaMallocManaged(&car_orig, sizeof(uint64_t));
  cudaMallocManaged(&ret_orig, sizeof(uint64_t));
  cudaMallocManaged(&car_ptx_1, sizeof(uint64_t));
  cudaMallocManaged(&ret_ptx_1, sizeof(uint64_t));
  cudaMallocManaged(&car_ptx_2, sizeof(uint64_t));
  cudaMallocManaged(&ret_ptx_2, sizeof(uint64_t));
  cudaMallocManaged(&car_ptx_3, sizeof(uint64_t));
  cudaMallocManaged(&ret_ptx_3, sizeof(uint64_t));

  // Create random data
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  a[0] = dis(gen);
  b[0] = dis(gen);
  c[0] = dis(gen);

  sxt_mac<<<1,1>>>(ret_orig, car_orig, a, b, c);
  sxt_mac_ptx_1<<<1,1>>>(ret_ptx_1, car_ptx_1, a, b, c);
  sxt_mac_ptx_2<<<1,1>>>(ret_ptx_2, car_ptx_2, a, b, c);
  sxt_mac_ptx_3<<<1,1>>>(ret_ptx_3, car_ptx_3, a, b, c);
  cudaDeviceSynchronize();

  // Print data
  std::cout << std::endl;
  std::cout << "ret_orig  = 0x" << std::hex << ret_orig[0] << std::endl;
  std::cout << "ret_ptx_1 = 0x" << std::hex << ret_ptx_1[0] << std::endl;
  std::cout << "ret_ptx_2 = 0x" << std::hex << ret_ptx_2[0] << std::endl;
  std::cout << "ret_ptx_3 = 0x" << std::hex << ret_ptx_3[0] << std::endl;
  std::cout << std::endl;
  std::cout << "car_orig  = 0x" << std::hex << car_orig[0] << std::endl;
  std::cout << "car_ptx_1 = 0x" << std::hex << car_ptx_1[0] << std::endl;
  std::cout << "car_ptx_2 = 0x" << std::hex << car_ptx_2[0] << std::endl;
  std::cout << "car_ptx_3 = 0x" << std::hex << car_ptx_3[0] << std::endl;
  std::cout << std::endl;

  return 0;
}
