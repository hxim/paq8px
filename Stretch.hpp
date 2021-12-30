#pragma once

#include <cinttypes>

/**
 * Inverse of @ref squash. d = ln(p/(1-p)), d scaled by 8 bits, p by 12 bits.
 * d has range -2047 to 2047 representing -8 to 8. p has range 0 to 4095.
 */

int stretch(int p);
