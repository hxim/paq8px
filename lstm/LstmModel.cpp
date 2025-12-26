#include "LstmModel.hpp"

LstmModel::LstmModel(const Shared* const sh)
  : shared(sh)
  , probs(alphabetSize)
  , apm1{ sh, 0x10000, 24, 255 }
  , apm2{ sh, 0x800, 24, 255 }
  , apm3{ sh, 1024, 24, 255 }
  , iCtx{ 11, 1, 9 }
  , top(alphabetSize - 1)
  , mid(0)
  , bot(0)
  , expected(0)
{
  float const init_value = 1.f / alphabetSize;
  for (size_t i = 0; i < alphabetSize; i++) {
    probs[i] = init_value;
  }
}
