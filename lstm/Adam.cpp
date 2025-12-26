#include "Adam.hpp"
#include "../Array.hpp"
#include <cmath>
#include <cstddef>

Adam::Adam(size_t length, float* w, float* g, float beta2, float epsilon)
  : length(length)
  , w(w)
  , g(g)
  , v(length)
  , beta2(beta2)
  , eps(epsilon)
{
}
