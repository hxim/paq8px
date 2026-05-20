#include "French.hpp"

bool French::isAbbreviation(Word *w) { return w->matchesAny(abbreviations, numAbbrev); }
