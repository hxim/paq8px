#pragma once

#include "LstmModel.hpp"
#include "SimdLstmModel.hpp"
#include "../Shared.hpp"
#include "../Utils.hpp"

template <std::size_t Bits = 8>
class LstmFactory {
public:
  static LstmModel<Bits>* CreateLSTM(
    const Shared* const sh,
    std::size_t const num_cells,
    std::size_t const num_layers,
    std::size_t const horizon,
    float const learning_rate,
    float const gradient_clip)
  {
    if (sh->chosenSimd == SIMDType::SIMD_AVX2)
      return new SIMDLstmModel<SIMDType::SIMD_AVX2, Bits>(sh, num_cells, num_layers, horizon, learning_rate, gradient_clip);
    else
      return new SIMDLstmModel<SIMDType::SIMD_NONE, Bits>(sh, num_cells, num_layers, horizon, learning_rate, gradient_clip);
  }
};
