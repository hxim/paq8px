#pragma once

#include <cstdint>
#include <cstdio>

class ByteModelToBitModel
{
public:
    void CalculateByteProbabilities(const float* const byteProbabilities, const size_t byteProbabilities_size);
    size_t GetExpectedByte(const float* const byteProbabilities, const size_t byteProbabilities_size) const;
    float p() const;
    void SliceForNextBit(const float* const byteProbabilities, int bit);

private:
    size_t lower{};
    size_t length{};
    float pSum0{};
    float pSum1{};
};
