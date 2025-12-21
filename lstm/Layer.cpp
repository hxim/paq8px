#include "Layer.hpp"
#include "SimdFunctions.hpp"
#include <cstring>

template<typename T>
Layer<T>::Layer(
    SIMDType simdType,
    size_t input_size,
    size_t auxiliary_input_size,
    size_t output_size,
    size_t num_cells,
    size_t horizon,
    bool useTanh,
    float beta2,
    float epsilon,
    float learningRate,
    float endLearningRate,
    float decayMultiplier,
    float powerNumerator,
    float powerDenominator,
    uint64_t decaySteps)
    : simd(simdType)
    , update(std::valarray<float>(input_size), num_cells)
    , v(std::valarray<float>(input_size), num_cells)
    , transpose(std::valarray<float>(num_cells), input_size - output_size - auxiliary_input_size)
    , norm(std::valarray<float>(num_cells), horizon)
    , inverse_variance(horizon)
    , gamma(1.f, num_cells)
    , gamma_u(num_cells)
    , gamma_v(num_cells)
    , beta(num_cells)
    , beta_u(num_cells)
    , beta_v(num_cells)
    , input_size(input_size)
    , auxiliary_input_size(auxiliary_input_size)
    , output_size(output_size)
    , num_cells(num_cells)
    , horizon(horizon)
    , learning_rate(0.f)
    , optimizer(simdType, beta2, epsilon)
    , activation_tanh(simdType)
    , activation_logistic(simdType)
    , decay(learningRate, endLearningRate, decayMultiplier, powerNumerator, powerDenominator, decaySteps)
    , use_tanh(useTanh)
    , weights(std::valarray<float>(input_size), num_cells)
    , state(std::valarray<float>(num_cells), horizon)
    , error(num_cells)
{
}

template<typename T>
void Layer<T>::ForwardPass(
    std::valarray<float> const& input,
    T const input_symbol,
    size_t const epoch)
{
    for (size_t i = 0; i < num_cells; i++) {
        if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
            norm[epoch][i] = dot256_ps_fma3(&input[0], &weights[i][output_size], input.size(), weights[i][input_symbol]);
#endif
        } else {
            float f = weights[i][input_symbol];
            for (size_t j = 0; j < input.size(); j++)
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
    
    if (use_tanh) {
        activation_tanh.Run(&state[epoch][0], num_cells);
    } else {
        activation_logistic.Run(&state[epoch][0], num_cells);
    }
}

template<typename T>
void Layer<T>::BackwardPass(
    std::valarray<float> const& input,
    std::valarray<float>* hidden_error,
    std::valarray<float>* stored_error,
    uint64_t const time_step,
    size_t const epoch,
    size_t const layer,
    T const input_symbol)
{
    if (epoch == horizon - 1) {
        memset(&gamma_u[0], 0, num_cells * sizeof(float));
        memset(&beta_u[0], 0, num_cells * sizeof(float));

        for (size_t i = 0; i < num_cells; i++) {
            memset(&update[i][0], 0, input_size * sizeof(float));
            size_t offset = output_size + auxiliary_input_size;
            for (size_t j = 0; j < transpose.size(); j++)
                transpose[j][i] = weights[i][j + offset];
        }
    }

    for (size_t i = 0; i < num_cells; i++) {
        beta_u[i] += error[i];
        gamma_u[i] += error[i] * norm[epoch][i];
        error[i] *= gamma[i] * inverse_variance[epoch];
    }

    float sop = SumOfProducts(&error[0], &norm[epoch][0], num_cells) / num_cells;
    for (size_t i = 0; i < num_cells; i++) 
        error[i] -= sop * norm[epoch][i];

    if (layer > 0) {
        for (size_t i = 0; i < num_cells; i++) {
            if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
                (*hidden_error)[i] += dot256_ps_fma3(&error[0], &transpose[num_cells + i][0], num_cells, 0.f);
#endif
            } else {
                float f = 0.f;
                for (size_t j = 0; j < num_cells; j++) {
                    f += error[j] * transpose[num_cells + i][j];
                }
                (*hidden_error)[i] += f;
            }
        }
    }

    if (epoch > 0) {
        for (size_t i = 0; i < num_cells; i++) {
            if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
                (*stored_error)[i] += dot256_ps_fma3(&error[0], &transpose[i][0], num_cells, 0.f);
#endif
            } else {
                float f = 0.f;
                for (size_t j = 0; j < num_cells; j++) {
                    f += error[j] * transpose[i][j];
                }
                (*stored_error)[i] += f;
            }
        }
    }

    std::slice slice = std::slice(output_size, input.size(), 1);
    for (size_t cell_index = 0; cell_index < num_cells; cell_index++) {
        update[cell_index][slice] += error[cell_index] * input;
        update[cell_index][input_symbol] += error[cell_index];
    }

    if (epoch == 0) {
        decay.Apply(learning_rate, time_step);
        for (size_t cell_index = 0; cell_index < num_cells; cell_index++)
            optimizer.Run(&update[cell_index], &v[cell_index], &weights[cell_index], learning_rate, time_step);
        optimizer.Run(&gamma_u, &gamma_v, &gamma, learning_rate, time_step);
        optimizer.Run(&beta_u, &beta_v, &beta, learning_rate, time_step);
    }
}

template<typename T>
void Layer<T>::Reset() {
    for (size_t i = 0; i < horizon; i++) {
        inverse_variance[i] = 0.f;
        for (size_t j = 0; j < num_cells; j++) {
            state[i][j] = 0.f;
            norm[i][j] = 0.f;
        }
    }
    
    for (size_t i = 0; i < num_cells; i++) {
        error[i] = 0.f;
        gamma[i] = 1.f;
        gamma_u[i] = 0.f;
        gamma_v[i] = 0.f;
        beta[i] = 0.f;
        beta_u[i] = 0.f;
        beta_v[i] = 0.f;
        for (size_t j = 0; j < input_size; j++) {
            update[i][j] = 0.f;
            v[i][j] = 0.f;
        }
    }
    
    for (size_t i = 0; i < transpose.size(); i++) {
        for (size_t j = 0; j < num_cells; j++)
            transpose[i][j] = 0.f;
    }
}

// Explicit template instantiation
template class Layer<uint8_t>;
template class Layer<uint16_t>;
template class Layer<uint32_t>;