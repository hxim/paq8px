#include "LstmFactory.hpp"

LstmModel* LstmFactory::CreateLSTM(
    const Shared* const sh,
    size_t const num_cells,
    size_t const num_layers,
    size_t const horizon,
    float const learning_rate)
{
    SIMDType simdType = sh->chosenSimd;
    
    if (simdType == SIMDType::SIMD_AVX2 || simdType == SIMDType::SIMD_AVX512) {
        return new SIMDLstmModel(sh, SIMDType::SIMD_AVX2, num_cells, num_layers, horizon, learning_rate);
    } else {
        return new SIMDLstmModel(sh, SIMDType::SIMD_NONE, num_cells, num_layers, horizon, learning_rate);
    }
}
