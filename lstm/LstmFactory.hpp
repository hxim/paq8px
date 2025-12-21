#pragma once

#include "LstmModel.hpp"
#include "SimdLstmModel.hpp"
#include "../Shared.hpp"
#include "../Utils.hpp"
#include "../SIMDType.hpp"

template <size_t Bits = 8>
class LstmFactory {
public:
    static LstmModel<Bits>* CreateLSTM(
        const Shared* const sh,
        size_t num_cells,
        size_t num_layers,
        size_t horizon,
        float learning_rate,
        float gradient_clip);
};