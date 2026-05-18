#include "English.hpp"

bool English::isAbbreviation(Word *w) { return w->matchesAny(abbreviations, numAbbrev); }
