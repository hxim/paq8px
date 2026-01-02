#pragma once

#include "../Utils.hpp"
#include "../Simd.hpp"
#include "../SimdType.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

class VectorFunctions
{
public:
  virtual float DotProduct(
    float const* x1,
    float const* x2,
    size_t const len
  ) = 0;

  virtual float SumOfSquares(
    float* array,
    size_t array_length
  ) = 0;

  virtual void NormalizeThenActivate_Sigmoid(
    size_t array_length,
    float* norm,
    float* state,
    float* gamma,
    float* beta,
    float inverse_variance
  ) = 0;

  virtual void NormalizeThenActivate_Tanh(
    size_t array_length,
    float* norm,
    float* state,
    float* gamma,
    float* beta,
    float inverse_variance
  ) = 0;

  virtual void BackpropagateErrors(
    size_t len,
    size_t base_offset,
    size_t hidden_size,
    float* weights,
    float* error,
    float* grad_store
  ) = 0;

  virtual float ComputeMaxLogit(
    float* result,
    size_t result_length
  ) = 0;

  virtual void MatvecThenSoftmax(
    float* hidden,
    float* logits,
    float* output_layer,
    float* output,
    size_t const hidden_size,
    size_t const output_size,
    size_t const output_offset
  ) = 0;

  virtual void Softmax(
    float* logits,
    float* probs,
    size_t len,
    float max_logit
  ) = 0;
};

constexpr float negative_infinity = -std::numeric_limits<float>::max();

// Scalar sigmoid and tanh
float sigmoid_pade_clipped(float x);
float tanh_pade_clipped(float x);

// Clipping constants
static constexpr float SIGMOID_CLIP_MIN = -5.95f;
static constexpr float SIGMOID_CLIP_MAX = +5.95f;
static constexpr float TANH_CLIP_MIN = SIGMOID_CLIP_MIN / 2.0f;
static constexpr float TANH_CLIP_MAX = SIGMOID_CLIP_MAX / 2.0f;


