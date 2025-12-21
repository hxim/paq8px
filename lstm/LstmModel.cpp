#include "LstmModel.hpp"

template <size_t Bits>
LstmModel<Bits>::LstmModel(const Shared* const sh)
    : shared(sh)
    , probs(1.f / Size, Size)
    , apm1{ sh, 0x10000, 24, 255 }
    , apm2{ sh, 0x800, 24, 255 }
    , apm3{ sh, 1024, 24, 255 }
    , iCtx{ 11, 1, 9 }
    , top(Size - 1)
    , mid(0)
    , bot(0)
    , expected(0)
{
}

// Explicit template instantiation for common sizes
template class LstmModel<8>;
template class LstmModel<16>;