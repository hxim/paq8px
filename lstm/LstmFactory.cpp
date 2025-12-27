#include "LstmFactory.hpp"

LstmModel* LstmFactory::CreateLSTM(
  const Shared* const sh,
  size_t const num_cells,      // 200
  size_t const num_layers,     // 2
  size_t const horizon,        // 100
  float const learning_rate)   // 0.06
{
  SIMDType simdType = sh->chosenSimd;

  if (simdType == SIMDType::SIMD_AVX2 || simdType == SIMDType::SIMD_AVX512) {
    return new SIMDLstmModel(
      sh,
      SIMDType::SIMD_AVX2,
      num_cells,           // 200
      num_layers,          // 2
      horizon,             // 100
      learning_rate);      // 0.06
  }
  else {
    return new SIMDLstmModel(
      sh,
      SIMDType::SIMD_NONE,
      num_cells,           // 200
      num_layers,          // 2
      horizon,             // 100
      learning_rate);      // 0.06
  }
}
