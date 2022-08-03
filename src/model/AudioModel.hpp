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
  uint32_t wMode = 0;
  explicit AudioModel(Shared* const sh);
  int s2(uint32_t i) const;
  int t2(uint32_t i) const;
  int x1(uint32_t i) const;
  int x2(uint32_t i) const;
  static int signedClip8(int i);
  static int signedClip16(int i);
};
