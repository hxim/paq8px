#include "Adam.hpp"
#include "../Simd.hpp"
#include <cmath>
#include <cstddef>

Adam::Adam(SIMDType simdType, float beta2Value, float epsilon)
    : simd(simdType)
    , beta2(beta2Value)
    , eps(epsilon)
{
}

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx")))
#endif
void Adam::RunSimdAVX(
    std::valarray<float>* g,
    std::valarray<float>* v,
    std::valarray<float>* w,
    float const learning_rate,
    uint64_t const time_step) const
{
    static constexpr size_t SIMDW = 8;
    static __m256 const vec_beta2 = _mm256_set1_ps(beta2);
    static __m256 const vec_eps = _mm256_set1_ps(eps);
    static __m256 const vec_beta2_complement = _mm256_set1_ps(1.f - beta2);

    double const t = static_cast<double>(time_step);
    float const bias_v = 1.f - static_cast<float>(std::pow(beta2, t));
    __m256 const vec_bias_v = _mm256_set1_ps(bias_v);
    __m256 const vec_lr = _mm256_set1_ps(learning_rate);
    size_t const len = g->size();
    size_t const limit = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW));
    size_t remainder = len & (SIMDW - 1);

    for (size_t i = 0; i < limit; i += SIMDW) {
        __m256 vec_gi = _mm256_loadu_ps(&(*g)[i]);
        __m256 vec_vi = _mm256_loadu_ps(&(*v)[i]);

        // v = beta2 * v + (1 - beta2) * g^2
        vec_vi = _mm256_mul_ps(vec_vi, vec_beta2);
        __m256 vec_gi_sq = _mm256_mul_ps(vec_gi, vec_gi);
        __m256 vec_term = _mm256_mul_ps(vec_gi_sq, vec_beta2_complement);
        vec_vi = _mm256_add_ps(vec_vi, vec_term);
        _mm256_storeu_ps(&(*v)[i], vec_vi);

        // scaled_gradient = g / (sqrt(v / bias_v) + eps)
        __m256 vec_v_corrected = _mm256_div_ps(vec_vi, vec_bias_v);
        __m256 vec_sqrt = _mm256_sqrt_ps(vec_v_corrected);
        __m256 vec_denom = _mm256_add_ps(vec_sqrt, vec_eps);
        __m256 vec_scaled_grad = _mm256_div_ps(vec_gi, vec_denom);

        // w = w - lr * scaled_gradient
        __m256 vec_wi = _mm256_loadu_ps(&(*w)[i]);
        __m256 vec_update = _mm256_mul_ps(vec_lr, vec_scaled_grad);
        vec_wi = _mm256_sub_ps(vec_wi, vec_update);
        _mm256_storeu_ps(&(*w)[i], vec_wi);
    }

    for (; remainder > 0; remainder--) {
        const size_t i = len - remainder;
        float g_val = (*g)[i];
        float v_old = (*v)[i];
        float g_sq = g_val * g_val;
        float v_new = v_old * beta2 + (1.f - beta2) * g_sq;
        (*v)[i] = v_new;
        float v_corrected = v_new / bias_v;
        float denom = std::sqrt(v_corrected) + eps;
        float scaled_gradient = g_val / denom;
        (*w)[i] = (*w)[i] - learning_rate * scaled_gradient;
    }
}

#endif

void Adam::RunSimdNone(
    std::valarray<float>* g,
    std::valarray<float>* v,
    std::valarray<float>* w,
    float const learning_rate,
    uint64_t const time_step) const
{
    float const t = static_cast<float>(time_step);
    float const bias_v = 1.f - std::pow(beta2, t);

    for (size_t i = 0; i < g->size(); i++) {
        float g_val = (*g)[i];
        float v_old = (*v)[i];
        float g_sq = g_val * g_val;
        float v_new = v_old * beta2 + (1.0f - beta2) * g_sq;
        (*v)[i] = v_new;
        float v_corrected = v_new / bias_v;
        float denom = std::sqrt(v_corrected) + eps;
        float scaled_gradient = g_val / denom;
        (*w)[i] = (*w)[i] - learning_rate * scaled_gradient;
    }
}

void Adam::Run(
    std::valarray<float>* g,
    std::valarray<float>* v,
    std::valarray<float>* w,
    float const learning_rate,
    uint64_t const time_step) const
{
    if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
        RunSimdAVX(g, v, w, learning_rate, time_step);
#endif
    } else {
        RunSimdNone(g, v, w, learning_rate, time_step);
    }
}