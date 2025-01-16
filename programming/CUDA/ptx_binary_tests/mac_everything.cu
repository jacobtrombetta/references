#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <utility>
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

//--------------------------------------------------------------------------------------------------
// mul_wide_limbs_old
//--------------------------------------------------------------------------------------------------
template <unsigned num_limbs>
static __device__ __forceinline__ void mul_wide_limbs_old(uint64_t* c, const uint64_t* f, const uint64_t* g) noexcept {
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
// mul_wide_limbs
//--------------------------------------------------------------------------------------------------
template <unsigned num_limbs>
__device__ __forceinline__ void mul_wide_limbs(uint64_t* c, const uint64_t* f, const uint64_t* g) noexcept {
/*
  const uint64_t* f = f_i;
  const uint64_t* g = g_i;
  uint64_t* c = c_i;
# pragma unroll
  for (unsigned i = 0; i < num_limbs; ++i) {
    addc_cc(0,0);
# pragma unroll
    for (unsigned j = 0; j < num_limbs; ++j) {
      c[i + j] = madc_lo_cc(f[i], g[j], c[i + j]);
    }
    c[i + num_limbs] = addc_cc(0,0);
    basn::cmov(c[i + num_limbs], uint64_t{0}, i == 0);
# pragma unroll
    for (unsigned j = 0; j < num_limbs; ++j) {
      c[i + j + 1] = madc_hi_cc(f[i], g[j], c[i + j + 1]);
    }


    // for (unsigned k = 0; k < i + num_limbs; ++k) {
    //    printf("c[%d] = 0x%lx\n", k, c[k]);
    // }
  }
  */

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
  sxt::basfld::wide_limbs result = sxt::basfld::mac_ptx_2(a[0], b[0], carry[0], c[0]);
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
// sxt_mul_wide_limbs_old
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mul_wide_limbs_old(uint64_t* __restrict__ ret,
                                       const uint64_t* __restrict__ a,
                                       const uint64_t* __restrict__ b) {
  sxt::basfld::mul_wide_limbs_old<4>(ret, a, b);
}

//--------------------------------------------------------------------------------------------------
// sxt_mul_wide_limbs
//--------------------------------------------------------------------------------------------------
__global__ void sxt_mul_wide_limbs(uint64_t* __restrict__ ret,
                                   const uint64_t* __restrict__ a,
                                   const uint64_t* __restrict__ b) {
  sxt::basfld::mul_wide_limbs<4>(ret, a, b);
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
// mean
//--------------------------------------------------------------------------------------------------
template <class T> static T mean(const std::vector<T>& data) {
  return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

//--------------------------------------------------------------------------------------------------
// std_dev
//--------------------------------------------------------------------------------------------------
template <class T> static T std_dev(const std::vector<T>& data) {
  const T m = mean(data);
  const T v = variance(data, m);
  return std::sqrt(v);
}

//--------------------------------------------------------------------------------------------------
// variance
//--------------------------------------------------------------------------------------------------
template <class T> 
static T variance(const std::vector<T>& data, T mean) {
  return std::accumulate(data.begin(), data.end(), 0.0,
                         [mean](T accumulator, T val) {
                           return accumulator + (val - mean) * (val - mean);
                         }) / 
                    (data.size() - 1);
}

//--------------------------------------------------------------------------------------------------
// t_value
//--------------------------------------------------------------------------------------------------
template <class T> 
static T t_value(const std::vector<T>& group1, const std::vector<T>& group2) {
  if (group1.size() != group2.size()) {
    throw std::invalid_argument("The two groups must have the same size");
  }

  T mean1 = mean(group1);
  T mean2 = mean(group2);
  T variance1 = variance(group1, mean1);
  T variance2 = variance(group2, mean2);
  
  const size_t size = group1.size();
  T pooledVariance = ((size - 1) * variance1 + (size - 1) * variance2) / (size + size - 2);
  T standardError = sqrt(2 * pooledVariance / size);
  
  return (mean1 - mean2) / standardError;
}

//--------------------------------------------------------------------------------------------------
// median
//--------------------------------------------------------------------------------------------------
template <class T> 
T median(const std::vector<T>& data) {
  std::vector<T> sorted_data = data;

  auto n = sorted_data.size() / 2;
  std::nth_element(sorted_data.begin(), sorted_data.begin() + n, sorted_data.end());

  if (sorted_data.size() % 2 == 0) {
    T max_of_lower_half = *std::max_element(sorted_data.begin(), sorted_data.begin() + n);
    return (max_of_lower_half + sorted_data[n]) / 2;
  } else {
    return sorted_data[n];
  }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main() {
  // Create data
  // uint64_t* a;
  // uint64_t* b;
  // uint64_t* car_orig;
  // uint64_t* ret_orig;
  // uint64_t* car_ptx_1;
  // uint64_t* ret_ptx_1;
  // uint64_t* car_ptx_2;
  // uint64_t* ret_ptx_2;
  // uint64_t* car_ptx_3;
  // uint64_t* ret_ptx_3;

  uint64_t* a;
  uint64_t* b;
  uint64_t* c;
  uint64_t* ret_wide_limbs_old;
  uint64_t* ret_wide_limbs;
  uint64_t* ret_wide_limbs_working;

  // Set parameters
  //constexpr unsigned n_elements = 1;
  //constexpr unsigned threads_per_block = 256;
  //constexpr unsigned blocks = (n_elements + threads_per_block - 1) / threads_per_block;

  // Copy data to device 
  // cudaMallocManaged(&a, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&b, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&car_orig, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&ret_orig, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&car_ptx_1, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&ret_ptx_1, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&car_ptx_2, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&ret_ptx_2, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&car_ptx_3, n_elements * sizeof(uint64_t));
  // cudaMallocManaged(&ret_ptx_3, n_elements * sizeof(uint64_t));

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

  sxt_mul_wide_limbs_old<<<1,1>>>(ret_wide_limbs_old, a, b);
  sxt_mul_wide_limbs<<<1,1>>>(ret_wide_limbs, a, b);
  sxt_mul_wide_4_limbs_unrolled<<<1,1>>>(ret_wide_limbs_working, a, b);
  cudaDeviceSynchronize();

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
  std::cout << " = ret_wide_limbs_old" << std::endl;

  std::cout << "0x";
  for (unsigned j = 2*num_limbs; j > 0; --j){
    std::cout << std::hex << " " << ret_wide_limbs[j-1];
  }
  std::cout << " = ret_wide_limbs" << std::endl;

  std::cout << "0x";
  for (unsigned j = 2*num_limbs; j > 0; --j){
    std::cout << std::hex << " " << ret_wide_limbs_working[j-1];
  }
  std::cout << " = ret_wide_limbs_og" << std::endl;

  //for (unsigned j = 0; j < 2*num_limbs; ++j) {
  //  std::cout << "ret_wide_limbs[" << j << "]    = 0x" << std::hex << ret_wide_limbs[j] << std::endl;
  //  std::cout << "ret_wide_limbs_og[" << j << "] = 0x" << std::hex << ret_wide_limbs_og[j] << std::endl;
  //}

  // Set parameters
  // unsigned repetitions = 1 << 20;
  // std::cout << "Benchmark MAC original to PTX over " << repetitions << " repetitions" << std::endl;

  /*
  // Test output match
  std::cout << "...test";
  const unsigned test_repetitions = 10;
  for (unsigned i = 0; i < test_repetitions; ++i) {
    for (unsigned j = 0; j < n_elements; ++j) {
      a[j] = dis(gen);
      b[j] = dis(gen);
      car_orig[j] = dis(gen);
      car_ptx_1[j] = car_orig[j];
      car_ptx_2[j] = car_orig[j];
      car_ptx_3[j] = car_orig[j];
    }
    sxt_mac<<<blocks, threads_per_block>>>(ret_orig, car_orig, a, b, n_elements, 1);
    sxt_mac_ptx_1<<<blocks, threads_per_block>>>(ret_ptx_1, car_ptx_1, a, b, n_elements, 1);
    sxt_mac_ptx_2<<<blocks, threads_per_block>>>(ret_ptx_2, car_ptx_2, a, b, n_elements, 1);
    sxt_mac_ptx_3<<<blocks, threads_per_block>>>(ret_ptx_3, car_ptx_3, a, b, n_elements, 1);
    cudaDeviceSynchronize();

    // Compare results
    for (unsigned j = 0; j < n_elements; ++j) {
      if (ret_orig[j] != ret_ptx_1[j] || car_orig[j] != car_ptx_1[j]) {
        std::cerr << " - Error: Output mismatch" << std::endl;
        return 1;
      }
      if (ret_orig[j] != ret_ptx_2[j] || car_orig[j] != car_ptx_2[j]) {
        std::cerr << " - Error: Output mismatch" << std::endl;
        return 1;
      }
      if (ret_orig[j] != ret_ptx_3[j] || car_orig[j] != car_ptx_3[j]) {
        std::cerr << " - Error: Output mismatch" << std::endl;
        return 1;
      }
    }
  }
  std::cout << " - PTX implementation output matches original over " << test_repetitions << " tests" << std::endl;

  // Warm up
  std::cout << "...Warm-up loop" << std::endl;
  for (unsigned j = 0; j < n_elements; ++j) {
    a[j] = dis(gen);
    b[j] = dis(gen);
    car_orig[j] = dis(gen);
    car_ptx_1[j] = car_orig[j];
    car_ptx_2[j] = car_orig[j];
    car_ptx_3[j] = car_orig[j];
  }
  sxt_mac<<<blocks, threads_per_block>>>(ret_orig, car_orig, a, b, n_elements, repetitions);
  sxt_mac_ptx_1<<<blocks, threads_per_block>>>(ret_ptx_1, car_ptx_1, a, b, n_elements, repetitions);
  sxt_mac_ptx_2<<<blocks, threads_per_block>>>(ret_ptx_2, car_ptx_2, a, b, n_elements, repetitions);
  sxt_mac_ptx_3<<<blocks, threads_per_block>>>(ret_ptx_3, car_ptx_3, a, b, n_elements, repetitions);
  cudaDeviceSynchronize();

  // Benchmark
  std::vector<double> times_sxt_mad;
  std::vector<double> times_sxt_mad_ptx_1;
  std::vector<double> times_sxt_mad_ptx_2;
  std::vector<double> times_sxt_mad_ptx_3;
  const unsigned n_benchmark_tests = 10;
  std::cout << "...Begin benchmark loop over " << n_benchmark_tests << " tests" << std::endl;
  for (unsigned test = 0; test <= n_benchmark_tests; ++test) {
    for (unsigned j = 0; j < n_elements; ++j) {
      a[j] = dis(gen);
      b[j] = dis(gen);
      car_orig[j] = dis(gen);
      car_ptx_1[j] = car_orig[j];
      car_ptx_2[j] = car_orig[j];
      car_ptx_3[j] = car_orig[j];
    }

    // Original
    auto start_time = std::chrono::steady_clock::now();
    sxt_mac<<<blocks, threads_per_block>>>(ret_orig, car_orig, a, b, n_elements, repetitions);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (test > 0) {
      times_sxt_mad.push_back(duration.count());
    }

    // PTX 1
    start_time = std::chrono::steady_clock::now();
    sxt_mac_ptx_1<<<blocks, threads_per_block>>>(ret_ptx_1, car_ptx_1, a, b, n_elements, repetitions);
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (test > 0) {
      times_sxt_mad_ptx_1.push_back(duration.count());
    }

    // PTX 2
    start_time = std::chrono::steady_clock::now();
    sxt_mac_ptx_2<<<blocks, threads_per_block>>>(ret_ptx_2, car_ptx_2, a, b, n_elements, repetitions);
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (test > 0) {
      times_sxt_mad_ptx_2.push_back(duration.count());
    }

    // PTX 3
    start_time = std::chrono::steady_clock::now();
    sxt_mac_ptx_3<<<blocks, threads_per_block>>>(ret_ptx_3, car_ptx_3, a, b, n_elements, repetitions);
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (test > 0) {
      times_sxt_mad_ptx_3.push_back(duration.count());
    }
  }
  */

  // Print results
  std::cout << "================ Results ================" << std::endl;
  //std::cout << " - mad   : " << median(times_sxt_mad) << " milliseconds" << std::endl;
  //std::cout << " - ptx 1 : " << median(times_sxt_mad_ptx_1) << " milliseconds" << std::endl;
  //std::cout << " - ptx 2 : " << median(times_sxt_mad_ptx_2) << " milliseconds" << std::endl;
  //std::cout << " - ptx 3 : " << median(times_sxt_mad_ptx_3) << " milliseconds" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << std::endl;

  std::cout << "================ T Value ================" << std::endl;
  //std::cout << " - mad vs ptx1  : " << t_value(times_sxt_mad, times_sxt_mad_ptx_1) << std::endl;
  //std::cout << " - mad vs ptx2  : " << t_value(times_sxt_mad, times_sxt_mad_ptx_2) << std::endl;
  //std::cout << " - mad vs ptx3  : " << t_value(times_sxt_mad, times_sxt_mad_ptx_3) << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << std::endl;

  return 0;
}
