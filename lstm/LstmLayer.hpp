#ifndef PAQ8PX_LSTMLAYER_HPP
#define PAQ8PX_LSTMLAYER_HPP

#include "Layer.hpp"
#include "Adam.hpp"
#include "Activations.hpp"
#include "PolynomialDecay.hpp"
#include "../utils.hpp"
#include "../SIMDType.hpp"
#include <vector>

template <SIMDType simd, typename T>
class LstmLayer {
private:
  std::valarray<float> state, state_error, stored_error;
  std::valarray<std::valarray<float>> tanh_state, input_gate_state, last_state;
  float gradient_clip;
  std::size_t num_cells, epoch, horizon, input_size, output_size;
  Layer<simd, Adam<simd, 25, 3, 9999, 4, 1, 6>, Logistic<simd>, PolynomialDecay<7, 3, 1, 3, 5, 4, 1, 2>, T> forget_gate; // initial learning rate: 7*10^-3; final learning rate = 1*10^-3; decay multiplier: 5*10^-4; power for decay: 1/2 (i.e. sqrt); Steps: 0
  Layer<simd, Adam<simd, 25, 3, 9999, 4, 1, 6>, Tanh<simd>,     PolynomialDecay<7, 3, 1, 3, 5, 4, 1, 2>, T> input_node;
  Layer<simd, Adam<simd, 25, 3, 9999, 4, 1, 6>, Logistic<simd>, PolynomialDecay<7, 3, 1, 3, 5, 4, 1, 2>, T> output_gate;
  void Clamp(std::valarray<float>* x) {
    for (std::size_t i = 0u; i < x->size(); i++)
      (*x)[i] = std::max<float>(std::min<float>(gradient_clip, (*x)[i]), -gradient_clip);
  }
  static ALWAYS_INLINE float Rand(float const range) {
    return ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * range;
  }
public:
  std::uint64_t update_steps;
  LstmLayer(
    std::size_t const input_size,
    std::size_t const auxiliary_input_size,
    std::size_t const output_size,
    std::size_t const num_cells,
    std::size_t const horizon,
    float const gradient_clip,
    float const learning_rate,
    float const range = 0.4f) :
    state(num_cells), state_error(num_cells), stored_error(num_cells),
    tanh_state(std::valarray<float>(num_cells), horizon),
    input_gate_state(std::valarray<float>(num_cells), horizon),
    last_state(std::valarray<float>(num_cells), horizon),
    gradient_clip(gradient_clip),
    num_cells(num_cells), epoch(0), horizon(horizon),
    input_size(auxiliary_input_size), output_size(output_size),
    forget_gate(input_size, auxiliary_input_size, output_size, num_cells, horizon),
    input_node(input_size, auxiliary_input_size, output_size, num_cells, horizon),
    output_gate(input_size, auxiliary_input_size, output_size, num_cells, horizon),
    update_steps(0u)
  {
    for (std::size_t i = 0u; i < num_cells; i++) {
      for (std::size_t j = 0u; j < forget_gate.weights[i].size(); j++) {
        forget_gate.weights[i][j] = Rand(range);
        input_node.weights[i][j] = Rand(range);
        output_gate.weights[i][j] = Rand(range);
      }
      forget_gate.weights[i][forget_gate.weights[i].size() - 1] = 1.f;
    }
  }

  void ForwardPass(
    std::valarray<float> const& input,
    T const input_symbol,
    std::valarray<float>* hidden,
    std::size_t const hidden_start)
  {
    last_state[epoch] = state;
    forget_gate.ForwardPass(input, input_symbol, epoch);
    input_node.ForwardPass(input, input_symbol, epoch);
    output_gate.ForwardPass(input, input_symbol, epoch);
    for (std::size_t i = 0u; i < num_cells; i++) {
      input_gate_state[epoch][i] = 1.0f - forget_gate.state[epoch][i];
      state[i] = state[i] * forget_gate.state[epoch][i] + input_node.state[epoch][i] * input_gate_state[epoch][i];
      tanh_state[epoch][i] = tanha(state[i]);
      (*hidden)[hidden_start + i] = output_gate.state[epoch][i] * tanh_state[epoch][i];
    }
    epoch++;
    if (epoch == horizon) epoch = 0u;
  }

  void BackwardPass(
    std::valarray<float> const& input,
    std::size_t const epoch,
    std::size_t const layer,
    T const input_symbol,
    std::valarray<float>* hidden_error)
  {
    for (std::size_t i = 0u; i < num_cells; i++) {
      if (epoch == horizon - 1u) {
        stored_error[i] = (*hidden_error)[i];
        state_error[i] = 0.0f;
      }
      else 
          stored_error[i] += (*hidden_error)[i];

      output_gate.error[i] = tanh_state[epoch][i] * stored_error[i] * output_gate.state[epoch][i] * (1.0f - output_gate.state[epoch][i]);
      state_error[i] += stored_error[i] * output_gate.state[epoch][i] * (1.0f - (tanh_state[epoch][i] * tanh_state[epoch][i]));
      input_node.error[i] = state_error[i] * input_gate_state[epoch][i] * (1.0f - (input_node.state[epoch][i] * input_node.state[epoch][i]));
      forget_gate.error[i] = (last_state[epoch][i] - input_node.state[epoch][i]) * state_error[i] * forget_gate.state[epoch][i] * input_gate_state[epoch][i];
      (*hidden_error)[i] = 0.0f;
      if (epoch > 0u) {
        state_error[i] *= forget_gate.state[epoch][i];
        stored_error[i] = 0.0f;
      }
    }
    if (epoch == 0u)
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
    for (std::size_t i = 0u; i < horizon; i++) {
      for (std::size_t j = 0u; j < num_cells; j++) {
        tanh_state[i][j] = 0.f;
        input_gate_state[i][j] = 0.f;
        last_state[i][j] = 0.f;
      }
    }
    for (std::size_t i = 0u; i < num_cells; i++) {
      state[i] = 0.f;
      state_error[i] = 0.f;
      stored_error[i] = 0.f;
    }
    epoch = 0u;
    update_steps = 0u;
  }

  std::vector<std::valarray<std::valarray<float>>*> Weights() {
    std::vector<std::valarray<std::valarray<float>>*> weights;
    weights.push_back(&forget_gate.weights);
    weights.push_back(&input_node.weights);
    weights.push_back(&output_gate.weights);
    return weights;
  }
};

#endif //PAQ8PX_LSTMLAYER_HPP
