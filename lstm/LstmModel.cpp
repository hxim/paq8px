#include "LstmModel.hpp"

LstmModel::LstmModel(const Shared* const sh)
    : shared(sh)
    , probs(1.f / alphabetSize, alphabetSize)
    , apm1{ sh, 0x10000, 24, 255 }
    , apm2{ sh, 0x800, 24, 255 }
    , apm3{ sh, 1024, 24, 255 }
    , iCtx{ 11, 1, 9 }
    , top(alphabetSize - 1)
    , mid(0)
    , bot(0)
    , expected(0)
{
}
