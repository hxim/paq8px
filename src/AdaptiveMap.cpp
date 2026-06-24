#include "AdaptiveMap.hpp"

AdaptiveMap::AdaptiveMap(const Shared* const sh, const int n) : shared(sh), t(n) {
  dt = DivisionTable::getDT();
}

// sum = n0+n1
// p1 = (n1+0.5) / (n0+n1+1) [Laplace smoothed]
//
// To update p1 given y (new bit seen):
// case: y=0
//  n0++ -> recalculate p1
// case: y=1
//  n1++ -> recalculate p1
// But calculating p1 this way involves a division by an arbitrary denominator which is slow.
// -> We will replace this division by a table-based multiplication by the reciprocal.
// To do so we switch from (n0,n1) to (sum,p1) which encodes the same information just in different form.
// This way the table becomes smaller as it depends on a single operand (sum) not two operands (n0, n1)
// Updating (sum,p1) is not as straightforward as before: sum++, but new_p1 = ?
// 
// We will see below that both update cases (y=0 and y=1) follow the same pattern:
//   new_p1 = old_p1 + delta
// where delta = (y - old_p1) / (sum+2)
// 
// Derivation:
//
// old_p1 = p1 = (n1+0.5) / (n0+n1+1) = (2*n1+1) / (2*(sum+1))      // multiply both numerator and denominator by 2
//
// update case: y=0
//   old_p1 = (2*n1+1) / (2*(sum+1))
//   new_p1 = (2*n1+1) / (2*(sum+2))                                 // n1 unchanged, sum -> sum+1
//          = old_p1 * (sum+1)/(sum+2)                               // multiply and divide by (2*(sum+1))
//          = old_p1 * (1 - 1/(sum+2))                               // rewrite (sum+1)/(sum+2) = 1 - 1/(sum+2)
//          = old_p1 + (0 - old_p1) / (sum+2)
//
// update case: y=1
//   old_p1 = (2*n1+1) / (2*(sum+1))
//   new_p1 = (2*n1+3) / (2*(sum+2))                                 // n1 -> n1+1, sum -> sum+1
//          = (2*n1+1)/(2*(sum+2)) + 1/(sum+2)                       // split numerator: (2*n1+3) = (2*n1+1) + 2
//          = old_p1 * (sum+1)/(sum+2) + 1/(sum+2)                   // multiply and divide first term by (2*(sum+1))
//          = old_p1 * (1 - 1/(sum+2)) + 1/(sum+2)                   // rewrite (sum+1)/(sum+2) = 1 - 1/(sum+2)
//          = old_p1 + (1 - old_p1) / (sum+2)
//
// That's:
//   new_p1 = old_p1 + (y - old_p1) / (sum+2)
// for both cases.
//
// With fixed-point arithmetic:
//
//-> delta_22bit = (target_22bit - p1_22bit) / (sum+2)
//               = (target_22bit - p1_22bit) * dt[sum] / (1<<30)     // where dt[sum] = (1<<30) / (sum+2)

void AdaptiveMap::update(uint32_t* const p, const int limit) {
  assert(limit > 0 && limit < 1024);
  uint32_t cell = p[0];
  const int sum = cell & 1023;      //sum=n0+n1
  const int64_t y = shared->State.y;
  const int64_t target_22bit = y << 22;
  const int64_t p1_22bit = cell >> 10;  //prediction
  const int delta_22bit = int(((target_22bit - p1_22bit) * int64_t(dt[sum])) >> 30); // see derivation above
  cell = (uint32_t(p1_22bit + delta_22bit) << 10) | (sum < limit ? sum + 1 : sum);
  p[0] = cell;
}
