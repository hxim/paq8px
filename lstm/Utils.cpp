#include "Utils.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

// Helper for bitcasting
float bitcast_u32_to_f32(uint32_t x) {
  float result;
  memcpy(&result, &x, sizeof(float));
  return result;
}

void CheckValue(float x, const char* const message) {
  if (std::isnan(x) || std::isinf(x)) {
    printf("NaN or Inf detected. That's not good. Value: %f\n%s\n", x, message);
    exit(1);
  }
}
