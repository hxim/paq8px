#pragma once

#include "Layer.hpp"
#include "Adam.hpp"
#include "Activations.hpp"
#include "PolynomialDecay.hpp"
#include "../Utils.hpp"
#include "../SIMDType.hpp"
#include <vector>

template <SIMDType simd, typename T>
class LstmLayer {
private:
  std::valarray<float> state;
  std::valarray<float> state_error;
  std::valarray<float> stored_error;
  
  std::valarray<std::valarray<float>> tanh_state;
  std::valarray<std::valarray<float>> input_gate_state;
  std::valarray<std::valarray<float>> last_state;
  
  float gradient_clip;
  
  size_t num_cells;
  size_t epoch;
  size_t horizon;
  size_t input_size;
  size_t output_size;

  Layer<simd, Adam<simd, 25, 3, 9999, 4, 1, 6>, Logistic<simd>, PolynomialDecay<7, 3, 1, 3, 5, 4, 1, 2>, T> forget_gate; // initial learning rate: 7*10^-3; final learning rate = 1*10^-3; decay multiplier: 5*10^-4; power for decay: 1/2 (i.e. sqrt); Steps: 0
  Layer<simd, Adam<simd, 25, 3, 9999, 4, 1, 6>, Tanh<simd>,     PolynomialDecay<7, 3, 1, 3, 5, 4, 1, 2>, T> input_node;
  Layer<simd, Adam<simd, 25, 3, 9999, 4, 1, 6>, Logistic<simd>, PolynomialDecay<7, 3, 1, 3, 5, 4, 1, 2>, T> output_gate;

  void Clamp(std::valarray<float>* x) {
    for (size_t i = 0; i < x->size(); i++)
      (*x)[i] = std::max<float>(std::min<float>(gradient_clip, (*x)[i]), -gradient_clip);
  }

  static ALWAYS_INLINE float Rand(float const range) {
    return ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * range;
  }
public:

  uint64_t update_steps;

  LstmLayer(
    size_t const input_size, //457
    size_t const auxiliary_input_size, //0
    size_t const output_size, //256
    size_t const num_cells, //200
    size_t const horizon, //100
    float const gradient_clip, //16.0
    float const range = 0.4f) :

    state(num_cells), 
    state_error(num_cells), 
    stored_error(num_cells),
    
    tanh_state(std::valarray<float>(num_cells), horizon),
    input_gate_state(std::valarray<float>(num_cells), horizon),
    last_state(std::valarray<float>(num_cells), horizon),

    gradient_clip(gradient_clip),

    num_cells(num_cells), 
    epoch(0), 
    horizon(horizon),
    input_size(auxiliary_input_size), 
    output_size(output_size),

    forget_gate(input_size, auxiliary_input_size, output_size, num_cells, horizon),
    input_node(input_size, auxiliary_input_size, output_size, num_cells, horizon),
    output_gate(input_size, auxiliary_input_size, output_size, num_cells, horizon),
    update_steps(0)
  {
    //set random weights for each 
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

  void ForwardPass(
    std::valarray<float> const& input,
    T const input_symbol,
    std::valarray<float>* hidden,
    size_t const hidden_start)
  {
    last_state[epoch] = state;

    forget_gate.ForwardPass(input, input_symbol, epoch);
    input_node.ForwardPass(input, input_symbol, epoch);
    output_gate.ForwardPass(input, input_symbol, epoch);

    for (size_t i = 0; i < num_cells; i++) {
      input_gate_state[epoch][i] = 1.0f - forget_gate.state[epoch][i];
      state[i] = state[i] * forget_gate.state[epoch][i] + input_node.state[epoch][i] * input_gate_state[epoch][i];
      tanh_state[epoch][i] = tanha(state[i]);
      (*hidden)[hidden_start + i] = output_gate.state[epoch][i] * tanh_state[epoch][i];
    }

    epoch++;
    if (epoch == horizon) epoch = 0;
  }

  void BackwardPass(
    std::valarray<float> const& input,
    size_t const epoch,
    size_t const layer,
    T const input_symbol,
    std::valarray<float>* hidden_error)
  {
    for (size_t i = 0; i < num_cells; i++) {

      if (epoch == horizon - 1) {
        stored_error[i] = (*hidden_error)[i];
        state_error[i] = 0.0f;
      }
      else {
        stored_error[i] += (*hidden_error)[i];
      }

      output_gate.error[i] = tanh_state[epoch][i] * stored_error[i] * output_gate.state[epoch][i] * (1.0f - output_gate.state[epoch][i]);
      state_error[i] += stored_error[i] * output_gate.state[epoch][i] * (1.0f - (tanh_state[epoch][i] * tanh_state[epoch][i]));
      input_node.error[i] = state_error[i] * input_gate_state[epoch][i] * (1.0f - (input_node.state[epoch][i] * input_node.state[epoch][i]));
      forget_gate.error[i] = (last_state[epoch][i] - input_node.state[epoch][i]) * state_error[i] * forget_gate.state[epoch][i] * input_gate_state[epoch][i];
      
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
    
    Clamp(&state_error);
    Clamp(&stored_error);
    Clamp(hidden_error);
  }

  void Reset() {
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

  std::vector<std::valarray<std::valarray<float>>*> Weights() {
    std::vector<std::valarray<std::valarray<float>>*> weights;
    weights.push_back(&forget_gate.weights);
    weights.push_back(&input_node.weights);
    weights.push_back(&output_gate.weights);
    return weights;
  }
};
