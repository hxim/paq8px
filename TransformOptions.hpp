#pragma once

struct TransformOptions
{
  bool skipRgb = false;
  bool useBruteForceDeflateDetection = false;
  TransformOptions(const Shared * const shared) {
    skipRgb = shared->GetOptionSkipRGB();
    useBruteForceDeflateDetection = shared->GetOptionBruteforceDeflateDetection();
  }
};
