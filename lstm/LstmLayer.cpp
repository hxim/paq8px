#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cassert>

#include "LstmLayer.hpp"
#include "SimdFunctions.hpp"

float LstmLayer::Rand(float const range) {
    return ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * range;
}

LstmLayer::LstmLayer(
    SIMDType simdType,
    size_t const input_size,
    size_t const output_size,
    size_t const num_cells,
    size_t const horizon,
    float const range)
    : simd(simdType)
    , state(num_cells)
    , state_error(num_cells)
    , stored_error(num_cells)
    , tanh_state(std::valarray<float>(num_cells), horizon)
    , input_gate_state(std::valarray<float>(num_cells), horizon)
    , last_state(std::valarray<float>(num_cells), horizon)
    , num_cells(num_cells)
    , epoch(0)
    , horizon(horizon)
    , forget_gate(simdType, input_size, output_size, num_cells, horizon,
                  false, 0.9999f, 1e-6f, 0.007f, 0.001f, 0.0005f, 1.0f, 2.0f, 0)
    , input_node(simdType, input_size, output_size, num_cells, horizon,
                 true, 0.9999f, 1e-6f, 0.007f, 0.001f, 0.0005f, 1.0f, 2.0f, 0)
    , output_gate(simdType, input_size, output_size, num_cells, horizon,
                  false, 0.9999f, 1e-6f, 0.007f, 0.001f, 0.0005f, 1.0f, 2.0f, 0)
    , update_steps(0)
{
    // Set random weights for each gate
    assert(input_size == forget_gate.weights[0].size());
    assert(input_size == input_node.weights[0].size());
    assert(input_size == output_gate.weights[0].size());
    
    for (size_t i = 0; i < num_cells; i++) {
        for (size_t j = 0; j < input_size; j++) {
            forget_gate.weights[i][j] = Rand(range);
            input_node.weights[i][j] = Rand(range);
            output_gate.weights[i][j] = Rand(range);
        }
        forget_gate.weights[i][input_size - 1] = 1.f; // bias
    }
}

void LstmLayer::ForwardPass(
    std::valarray<float> const& input,
    uint8_t const input_symbol,
    std::valarray<float>* hidden,
    size_t const hidden_start)
{
    last_state[epoch] = state;

    forget_gate.ForwardPass(input, input_symbol, epoch);
    input_node.ForwardPass(input, input_symbol, epoch);
    output_gate.ForwardPass(input, input_symbol, epoch);

    for (size_t i = 0; i < num_cells; i++) {
        input_gate_state[epoch][i] = 1.0f - forget_gate.state[epoch][i];
        state[i] = state[i] * forget_gate.state[epoch][i] + 
                   input_node.state[epoch][i] * input_gate_state[epoch][i];

        tanh_state[epoch][i] = tanh_pade_clipped(state[i]);
        (*hidden)[hidden_start + i] = output_gate.state[epoch][i] * tanh_state[epoch][i];
    }

    epoch++;
    if (epoch == horizon) 
        epoch = 0;
}

void LstmLayer::BackwardPass(
    std::valarray<float> const& input,
    size_t const epoch,
    size_t const layer,
    uint8_t const input_symbol,
    std::valarray<float>* hidden_error)
{
    for (size_t i = 0; i < num_cells; i++) {
        if (epoch == horizon - 1) {
            stored_error[i] = (*hidden_error)[i];
            state_error[i] = 0.0f;
        } else {
            stored_error[i] += (*hidden_error)[i];
        }

        output_gate.error[i] = tanh_state[epoch][i] * stored_error[i] * 
                               output_gate.state[epoch][i] * (1.0f - output_gate.state[epoch][i]);
        state_error[i] += stored_error[i] * output_gate.state[epoch][i] * 
                         (1.0f - (tanh_state[epoch][i] * tanh_state[epoch][i]));
        input_node.error[i] = state_error[i] * input_gate_state[epoch][i] * 
                             (1.0f - (input_node.state[epoch][i] * input_node.state[epoch][i]));
        forget_gate.error[i] = (last_state[epoch][i] - input_node.state[epoch][i]) * 
                              state_error[i] * forget_gate.state[epoch][i] * input_gate_state[epoch][i];
        
        (*hidden_error)[i] = 0.0f;
        
        if (epoch > 0) {
            state_error[i] *= forget_gate.state[epoch][i];
            stored_error[i] = 0.0f;
        }
    }

    if (epoch == 0)
        update_steps++;

    forget_gate.BackwardPass(input, hidden_error, &stored_error, update_steps, epoch, layer, input_symbol);
    input_node.BackwardPass(input, hidden_error, &stored_error, update_steps, epoch, layer, input_symbol);
    output_gate.BackwardPass(input, hidden_error, &stored_error, update_steps, epoch, layer, input_symbol);
}

void LstmLayer::Reset() {
    forget_gate.Reset();
    input_node.Reset();
    output_gate.Reset();

    for (size_t i = 0; i < horizon; i++) {
        for (size_t j = 0; j < num_cells; j++) {
            tanh_state[i][j] = 0.f;
            input_gate_state[i][j] = 0.f;
            last_state[i][j] = 0.f;
        }
    }

    for (size_t i = 0; i < num_cells; i++) {
        state[i] = 0.f;
        state_error[i] = 0.f;
        stored_error[i] = 0.f;
    }

    epoch = 0;
    update_steps = 0;
}

std::vector<std::valarray<std::valarray<float>>*> LstmLayer::Weights() {
    std::vector<std::valarray<std::valarray<float>>*> weights;
    weights.push_back(&forget_gate.weights);
    weights.push_back(&input_node.weights);
    weights.push_back(&output_gate.weights);
    return weights;
}
