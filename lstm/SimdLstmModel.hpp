#pragma once

#include "LstmModel.hpp"
#include "Lstm.hpp"
#include "SimdFunctions.hpp"
#include "../APM.hpp"
#include "../BlockType.hpp"
#include "../IndirectContext.hpp"
#include "../Mixer.hpp"
#include "../RingBuffer.hpp"
#include "../Shared.hpp"
#include "../SIMDType.hpp"

template <size_t Bits = 8>
class SIMDLstmModel : public LstmModel<Bits> {
private:  
    SIMDType simd;
    LSTM::Shape shape;
    Lstm<uint8_t> lstm;
    LSTM::Repository repo;
    LSTM::Model::Type modelType, pModelType;
    BlockType pBlockType;

public:
    SIMDLstmModel(
        const Shared* const sh,
        SIMDType simdType,
        size_t num_cells,
        size_t num_layers,
        size_t horizon,
        float learning_rate,
        float gradient_clip);

    void mix(Mixer& m) override;
};