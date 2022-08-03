#include "German.hpp"

bool German::isAbbreviation(Word *w) { return w->matchesAny(abbreviations, numAbbrev); }
