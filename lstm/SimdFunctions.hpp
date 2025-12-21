#pragma once

#include "../Utils.hpp"
#include "../Simd.hpp"
#include "../SimdType.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>

// Forward declarations for activation classes
class Tanh;
class Logistic;

// SIMD horizontal sum functions
#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__)) 
__attribute__((target("sse3")))
#endif
float hsum_ps_sse3(__m128 v);

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx")))
#endif
float hsum256_ps_avx(__m256 v);

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
float dot256_ps_avx2(float const* x1, float const* x2, size_t len, float init);

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
float sum256_ps_avx2(float const* x, size_t len, float init);

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
void softmax_avx2(float* logits, float* probs, size_t len, float max_logit);

#endif // X64_SIMD_AVAILABLE

// Non-SIMD vector functions
float SumOfSquares(float* v1, size_t n);
float SumOfProducts(float* v1, float* v2, size_t n);

// Softmax
void softmax_scalar(float* logits, float* probs, size_t len, float max_logit);

// Scalar tanh
float tanh_pade_clipped(float x);

// Activation function classes
class Tanh {
public:
  explicit Tanh(SIMDType simdType);
  void Run(float* f, size_t len) const;

private:
  SIMDType simd;
#ifdef X64_SIMD_AVAILABLE
  void RunSimdAVX(float* f, size_t len) const;
#endif
};

class Logistic {
public:
  explicit Logistic(SIMDType simdType);
  void Run(float* f, size_t len) const;

private:
  SIMDType simd;
#ifdef X64_SIMD_AVAILABLE
  void RunSimdAVX(float* f, size_t len) const;
#endif
};

