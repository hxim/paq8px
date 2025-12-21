#pragma once

#include "../Utils.hpp"
#include "../SIMDType.hpp"
#include <cstdint>
#include <valarray>

class Adam {
private:
    SIMDType simd;
    float beta2;
    float eps;

#ifdef X64_SIMD_AVAILABLE
    void RunSimdAVX(
        std::valarray<float>* g,
        std::valarray<float>* v,
        std::valarray<float>* w,
        float learning_rate,
        uint64_t time_step) const;
#endif

    void RunSimdNone(
        std::valarray<float>* g,
        std::valarray<float>* v,
        std::valarray<float>* w,
        float learning_rate,
        uint64_t time_step) const;

public:
    Adam(SIMDType simdType, float beta2Value, float epsilon);
    ~Adam() = default;
    
    void Run(
        std::valarray<float>* g,
        std::valarray<float>* v,
        std::valarray<float>* w,
        float learning_rate,
        uint64_t time_step) const;
};