#include "Predictor.hpp"

Predictor::Predictor(Shared* const sh) : shared(sh) { shared->reset(); }
