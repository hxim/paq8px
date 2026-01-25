#include "Adam.hpp"
#include "../Array.hpp"
#include <cmath>
#include <cstddef>

Adam::Adam(size_t length, float* w, float* g, float base_lr)
  : length(length)
  , w(w)
  , g(g)
  , v(length)
  , base_lr(base_lr)
{
}
