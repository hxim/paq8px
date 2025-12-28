#include "ByteModelToBitModel.hpp"
#include <cmath>

static void MakeSum(const float* const byteProbabilities, float* sum0, float* sum1, size_t length)
{
    float pSum0 = 0.0f;
    float pSum1 = 0.0f;
    for (size_t i = 0; i < length; i++)
    {
        pSum0 += byteProbabilities[i];
        pSum1 += byteProbabilities[length + i];
    }
    *sum0 = pSum0;
    *sum1 = pSum1;
}

void ByteModelToBitModel::CalculateByteProbabilities(const float* const byteProbabilities, const size_t byteProbabilities_size)
{
    lower = 0;
    length = byteProbabilities_size >> 1;
    MakeSum(byteProbabilities + lower, &pSum0, &pSum1, length);
}

size_t ByteModelToBitModel::GetExpectedByte(const float* const byteProbabilities, const size_t byteProbabilities_size) const
{
    float largest = byteProbabilities[0];
    size_t largest_idx = 0;
    for (size_t i = 1; i < byteProbabilities_size; i++)
    {
        if (largest < byteProbabilities[i])
        {
            largest_idx = i;
            largest = byteProbabilities[i];
        }
    }
    return largest_idx;
}

float ByteModelToBitModel::p() const
{
    if (pSum0 <= 0.0f && pSum1 <= 0.0f)
    {
        return 0.5f;
    }
    float p = pSum1 / (pSum0 + pSum1);
    return p;
}

void ByteModelToBitModel::SliceForNextBit(const float* const byteProbabilities, int bit)
{
    if (bit == 1)
        lower += length;
    length >>= 1;
    if (length == 0)
        return; //no next bit
    MakeSum(byteProbabilities + lower, &pSum0, &pSum1, length);
}
