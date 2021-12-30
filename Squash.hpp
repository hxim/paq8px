#pragma once

/**
 * return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
 * @param d
 * @return
 */

int squash(int d);
