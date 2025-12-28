#pragma once

#include "LstmModel.hpp"
#include "../Shared.hpp"
#include "../Utils.hpp"
#include "../SIMDType.hpp"

class LstmFactory {
public:
  static LstmModel* CreateLSTM(
    const Shared* const sh,
    size_t num_cells,
    size_t num_layers,
    size_t horizon,
    float learning_rate);
};
