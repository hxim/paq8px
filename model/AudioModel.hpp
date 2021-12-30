#pragma once

#include "../Shared.hpp"
#include <cstdint>

/**
 * Model a 16/8-bit stereo/mono uncompressed .wav file.
 * Based on 'An asymptotically Optimal Predictor for Stereo Lossless Audio Compression' by Florin Ghido.
 * Base class for common functions of the 16-bit and 8-bit audio models.
 */
class AudioModel {
protected:
    Shared * const shared;
    int s = 0;
    int wMode = 0;
    explicit AudioModel(Shared* const sh);
    auto s2(int i) -> int;
    auto t2(int i) -> int;
    auto x1(int i) -> int;
    auto x2(int i) -> int;
    static auto signedClip8(int i) -> int;
    static auto signedClip16(int i) -> int;
};
