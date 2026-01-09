#include "Adam.hpp"
#include "../Array.hpp"
#include <cmath>
#include <cstddef>

Adam::Adam(size_t length, float* w, float* g, float base_lr, float beta2, float epsilon)
  : length(length)
  , w(w)
  , g(g)
  , v(length)
  , base_lr(base_lr)
  , beta2(beta2)
  , eps(epsilon)
{
}
