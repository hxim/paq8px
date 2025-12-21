#include "LstmFactory.hpp"

template <size_t Bits>
LstmModel<Bits>* LstmFactory<Bits>::CreateLSTM(
    const Shared* const sh,
    size_t const num_cells,
    size_t const num_layers,
    size_t const horizon,
    float const learning_rate,
    float const gradient_clip)
{
    SIMDType simdType = sh->chosenSimd;
    
    if (simdType == SIMDType::SIMD_AVX2 || simdType == SIMDType::SIMD_AVX512) {
        return new SIMDLstmModel<Bits>(sh, SIMDType::SIMD_AVX2, num_cells, num_layers, horizon, learning_rate, gradient_clip);
    } else {
        return new SIMDLstmModel<Bits>(sh, SIMDType::SIMD_NONE, num_cells, num_layers, horizon, learning_rate, gradient_clip);
    }
}

// Explicit template instantiation
template class LstmFactory<8>;
template class LstmFactory<16>;