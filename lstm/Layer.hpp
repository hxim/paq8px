#ifndef PAQ8PX_LAYER_HPP
#define PAQ8PX_LAYER_HPP

#include "SimdFunctions.hpp"
#include "IOptimizer.hpp"
#include "IActivation.hpp"
#include "IDecay.hpp"
#include "../utils.hpp"

template<SIMDType simd, class Optimizer, class Activation, class Decay, typename T>
class Layer {
  static_assert(std::is_base_of<IOptimizer, Optimizer>::value, "Optimizer must implement IOptimizer interface");
  static_assert(std::is_base_of<IActivation, Activation>::value, "Activation must implement IActivation interface");
  static_assert(std::is_base_of<IDecay, Decay>::value, "Decay must implement IDecay interface");
private:
  std::valarray<std::valarray<float>> update, m, v, transpose, norm;
  std::valarray<float> inverse_variance;
  std::valarray<float> gamma, gamma_u, gamma_m, gamma_v;
  std::valarray<float> beta, beta_u, beta_m, beta_v;
  std::size_t input_size, auxiliary_input_size, output_size, num_cells, horizon;
  float learning_rate;
  Optimizer optimizer;
  Activation activation;
  Decay decay;
public:
  std::valarray<std::valarray<float>> weights, state;
  std::valarray<float> error;
  Layer(
    std::size_t const input_size,
    std::size_t const auxiliary_input_size,
    std::size_t const output_size,
    std::size_t const num_cells,
    std::size_t const horizon
  ) :
    update(std::valarray<float>(input_size), num_cells),
    m(std::valarray<float>(input_size), num_cells),
    v(std::valarray<float>(input_size), num_cells),
    transpose(std::valarray<float>(num_cells), input_size - output_size - auxiliary_input_size),
    norm(std::valarray<float>(num_cells), horizon),
    inverse_variance(horizon),
    gamma(1.f, num_cells), gamma_u(num_cells), gamma_m(num_cells), gamma_v(num_cells),
    beta(num_cells), beta_u(num_cells), beta_m(num_cells), beta_v(num_cells),
    input_size(input_size), auxiliary_input_size(auxiliary_input_size), output_size(output_size), num_cells(num_cells), horizon(horizon),
    learning_rate(0.f),
    weights(std::valarray<float>(input_size), num_cells),
    state(std::valarray<float>(num_cells), horizon),
    error(num_cells)
  {};

  void ForwardPass(
    std::valarray<float> const& input,
    T const input_symbol,
    std::size_t const epoch)
  {
    for (std::size_t i = 0; i < num_cells; i++) {
      if (simd == SIMD_AVX2)
        norm[epoch][i] = dot256_ps_fma3(&input[0], &weights[i][output_size], input.size(), weights[i][input_symbol]);
      else {
        float f = weights[i][input_symbol];
        for (std::size_t j = 0; j < input.size(); j++)
          f += input[j] * weights[i][output_size + j];
        norm[epoch][i] = f;
      }
    }
    const float ss = SumOfSquares(&norm[epoch][0], num_cells);
    inverse_variance[epoch] = 1.f / std::sqrt(ss / num_cells + 1e-5f);
    for (size_t i = 0; i < num_cells; i++) {
      norm[epoch][i] *= inverse_variance[epoch];
      state[epoch][i] = norm[epoch][i] * gamma[i] + beta[i];
    }
    activation.Run(&state[epoch][0], num_cells);
  };

  void BackwardPass(
    std::valarray<float> const& input,
    std::valarray<float>* hidden_error,
    std::valarray<float>* stored_error,
    std::uint64_t const time_step,
    std::size_t const epoch,
    std::size_t const layer,
    T const input_symbol)
  {
    if (epoch == horizon - 1) {
       memset(&gamma_u[0], 0, num_cells * sizeof(float));
       memset(&beta_u[0], 0, num_cells * sizeof(float));
      for (std::size_t i = 0; i < num_cells; i++) {
          memset(&update[i][0], 0, input_size * sizeof(float));
        std::size_t offset = output_size + auxiliary_input_size;
        for (std::size_t j = 0; j < transpose.size(); j++)
          transpose[j][i] = weights[i][j + offset];
      }
    }
    for (size_t i = 0; i < num_cells; i++) {
      beta_u[i] += error[i];
      gamma_u[i] += error[i] * norm[epoch][i];
      error[i] *= gamma[i] * inverse_variance[epoch];
    }
    float sop = SumOfProducts(&error[0], &norm[epoch][0], num_cells) / num_cells;
    for (size_t i = 0; i < num_cells; i++) error[i] -= sop*norm[epoch][i];

    if (layer > 0) {
      for (std::size_t i = 0; i < num_cells; i++) {
        if (simd == SIMD_AVX2)
          (*hidden_error)[i] += dot256_ps_fma3(&error[0], &transpose[num_cells + i][0], num_cells, 0.f);
        else {
          float f = 0.f;
          for (std::size_t j = 0; j < num_cells; j++)
            f += error[j] * transpose[num_cells + i][j];
          (*hidden_error)[i] += f;
        }
      }
    }
    if (epoch > 0) {
      for (std::size_t i = 0; i < num_cells; i++) {
        if (simd == SIMD_AVX2)
          (*stored_error)[i] += dot256_ps_fma3(&error[0], &transpose[i][0], num_cells, 0.f);
        else {
          float f = 0.f;
          for (std::size_t j = 0; j < num_cells; j++)
            f += error[j] * transpose[i][j];
          (*stored_error)[i] += f;
        }
      }
    }
    decay.Apply(learning_rate, time_step);
    std::slice slice = std::slice(output_size, input.size(), 1);
    for (std::size_t i = 0; i < num_cells; i++) {
      update[i][slice] += error[i] * input;
      update[i][input_symbol] += error[i];
    }
    if (epoch == 0) {
      for (std::size_t i = 0; i < num_cells; i++)
        optimizer.Run(&update[i], &m[i], &v[i], &weights[i], learning_rate, time_step);
      optimizer.Run(&gamma_u, &gamma_m, &gamma_v, &gamma, learning_rate, time_step);
      optimizer.Run(&beta_u, &beta_m, &beta_v, &beta, learning_rate, time_step);
    }
  }

  void Reset() {
    for (std::size_t i = 0u; i < horizon; i++) {
      inverse_variance[i] = 0.f;
      for (std::size_t j = 0u; j < num_cells; j++) {
        state[i][j] = 0.f;
        norm[i][j] = 0.f;
      }
    }
    for (std::size_t i = 0u; i < num_cells; i++) {
      error[i] = 0.f;
      gamma[i] = 1.f, gamma_u[i] = 0.f, gamma_m[i] = 0.f, gamma_v[i] = 0.f;
      beta[i] = 0.f, beta_u[i] = 0.f, beta_m[i] = 0.f, beta_v[i] = 0.f;
      for (std::size_t j = 0u; j < input_size; j++) {
        update[i][j] = 0.f;
        m[i][j] = 0.f;
        v[i][j] = 0.f;
      }
    }
    for (std::size_t i = 0u; i < transpose.size(); i++) {
      for (std::size_t j = 0u; j < num_cells; j++)
        transpose[i][j] = 0.f;
    }
  }
};

#endif //PAQ8PX_LAYER_HPP
