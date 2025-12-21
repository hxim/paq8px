#pragma once

#include "../Utils.hpp"
#include "../SIMDType.hpp"
#include <cstddef>

class Tanh {
private:
    SIMDType simd;

#ifdef X64_SIMD_AVAILABLE
    void RunSimdAVX2(float* f, size_t len) const;
#endif

public:
    explicit Tanh(SIMDType simdType);
    ~Tanh() = default;
    
    void Run(float* f, size_t len) const;
};

class Logistic {
private:
    SIMDType simd;

#ifdef X64_SIMD_AVAILABLE
    void RunSimdAVX2(float* f, size_t len) const;
#endif

public:
    explicit Logistic(SIMDType simdType);
    ~Logistic() = default;
    
    void Run(float* f, size_t len) const;
};