#pragma once

#include "IPredictor.hpp"
#include "Bucket16.hpp"
#include "HashElementForContextMap.hpp"
#include "Hash.hpp"
#include "Ilog.hpp"
#include "Mixer.hpp"
#include "Random.hpp"
#include "StateMap.hpp"
#include "StateTable.hpp"
#include "Stretch.hpp"

/////////////////////////// ContextMap /////////////////////////
// TODO(epsteina): update this documentation
// A ContextMap maps contexts to bit histories and makes predictions
// to a Mixer.  Methods common to all classes:
//
// ContextMap cm(m, c); creates using about m bytes of memory (a power
//   of 2) for c contexts.
// cm.set(cx);  sets the next context to cx, called up to c times
//   cx is an arbitrary 32 bit value that identifies the context.
//   It should be called before predicting the first bit of each byte.
// cm.mix(m) updates Mixer m with the next prediction.  Returns 1
//   if context cx is found, else 0.  Then it extends all the contexts with
//   global bit y.  It should be called for every bit:
//
//     if (bitPosition==0)
//       for (int i=0; i<c; ++i) cm.set(cxt[i]);
//     cm.mix(m);
//
// The different types are as follows:
//

// - SmallStationaryContextMap.  0 <= cx*256 < m.
//     The state is a 16-bit probability that is adjusted after each
//     prediction.  c=1.
// - ContextMap.  For large contexts, c >= 1.  context must be hashed.

// context map for large contexts.  Most modeling uses this type of context
// map.  It includes a built in RunContextMap to predict the last byte seen
// in the same context, and also bit-level contexts that map to a bit
// history state.
//
// Bit histories are stored in a hash table.  The table is organized into
// 64-byte buckets aligned on cache page boundaries.  Each bucket contains
// a hash chain of 7 elements, plus a 2 element queue (packed into 1 byte)
// of the last 2 elements accessed for LRU replacement.  Each element has
// a 2 byte checksum for detecting collisions, and an array of 7 bit history
// states indexed by the last 0 to 2 bits of context.  The buckets are indexed
// by a context ending after 0, 2, or 5 bits of the current byte.  Thus, each
// byte modeled results in 3 main memory accesses per context, with all other
// accesses to cache.
//
// On bits 0, 2 and 5, the context is updated and a new bucket is selected.
// The most recently accessed element is tried first, by comparing the
// 16 bit checksum, then the 7 elements are searched linearly.  If no match
// is found, then the element with the lowest priority among the 5 elements
// not in the LRU queue is replaced.  After a replacement, the queue is
// emptied (so that consecutive misses favor a LFU replacement policy).
// In all cases, the found/replaced element is put in the front of the queue.
//
// The priority is the state number of the first element (the one with 0
// additional bits of context).  The states are sorted by increasing n0+n1
// (number of bits seen), implementing a LFU replacement policy.
//
// When the context ends on a byte boundary (bit 0), only 3 of the 7 bit
// history states are used.  The remaining 4 bytes implement a run model
// as follows: <count:7,d:1> <b1> <unused> <unused> where <b1> is the last byte
// seen, possibly repeated.  <count:7,d:1> is a 7 bit count and a 1 bit
// flag (represented by count * 2 + d).  If d=0 then <count> = 1..127 is the
// number of repeats of <b1> and no other bytes have been seen.  If d is 1 then
// other byte values have been seen in this context prior to the last <count>
// copies of <b1>.
//
// As an optimization, the last two hash elements of each byte (representing
// contexts with 2-7 bits) are not updated until a context is seen for
// a second time.  This is indicated by <count,d> = <1,0> (2).  After update,
// <count,d> is updated to <2,0> or <1,1> (4 or 3).

class ContextMap : IPredictor {
public:
  static constexpr int MIXERINPUTS = 5;

private:
  const Shared * const shared;
  Random rnd;
  const int C; /**< max number of contexts */
  Array<Bucket16<HashElementForContextMap, 7>> t; /**< bit and byte histories (statistics) */
  Array<uint8_t *> cp; /**< @ref C pointers to current bit history */
  Array<uint32_t> cxt; /**< c whole byte context hashes */
  Array<uint16_t> chk; /**< c whole byte context checksums */
  Array<uint8_t *> runP; /**< c [0..3] = count, value, unused, unused */
  StateMap sm; /**< c maps of state -> p */
  int cn; /**< next context to set by set() */
  const uint32_t mask;
  const int hashBits;
  uint64_t validFlags;
  Ilog *ilog = &Ilog::getInstance();

public:
  /**
    * Construct using @ref m bytes of memory for @ref contexts contexts
    * @param m bytes of memory to use
    * @param contexts max number of contexts
    */
  ContextMap(const Shared* const sh, uint64_t m, int contexts);

  /**
    * Set next whole byte context to @ref cx.
    * @param cx
    */
  void set(uint64_t cx);
  void skip();
  void update() override;
  void mix(Mixer &m);
};
