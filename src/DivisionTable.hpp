#pragma once

/**
 * This class provides a static (common) 1024-element lookup table for integer division
 * Initialization will run multiple times, but the table is created only once
 * @todo Split into declaration and definition
 */
class DivisionTable {
public:
  static int* getDT() {
    static int dt[1024];
    for( int n = 0; n < 1024; ++n ) {
      dt[n] = (1 << 30) / (n + 2); // 1/(n+2) scaled by 30 bits
    }
    return dt;
  }
};
