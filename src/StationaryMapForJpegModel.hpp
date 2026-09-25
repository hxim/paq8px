#pragma once

#include "IPredictor.hpp"
#include "Hash.hpp"
#include "Mixer.hpp"
#include "Random.hpp"
#include "Stretch.hpp"
#include "UpdateBroadcaster.hpp"
#include "Utils.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>

/**
 * A bank of maps for contexts whose statistics change slowly (nearly stationary data).
 * For each bit, it counts how many 0s and 1s were seen in the same context, and
 * uses those counts to predict the next bit.
 *
 * One bank serves numContexts contexts from a single shared hash table.
 *
 * Terms used below:
 *
 *   element  The statistics of one context: a small bit tree of
 *            (2^InputBits)-1 counters (each a 16 bit n0 and a 16 bit n1),
 *            plus a 24 bit checksum that identifies the context.
 *   group    ElementsPerGroup elements that share one hash table bucket.
 *            A context may be stored in any element of its group.
 *   chunk    The InputBits bits that are modelled with one context, i.e.
 *            the bits between two calls of set().
 *
 * A group is 256 bytes (four cache lines). Everything needed to find the right
 * element is in the first cache line:
 *
 *   line 0    chkLo[ElementsPerGroup]     low 16 bits of each checksum, 0 = empty element
 *             root[ElementsPerGroup]      counters[0] (the root of the bit tree) of each element
 *             order                       the ranking of the elements, one byte per rank
 *             chkHi[ElementsPerGroup]     high 8 bits of each checksum
 *   line 1-3  tail[ElementsPerGroup][...] counters[1..] of each element
 *
 * So searching the group, comparing checksums and choosing an element to
 * replace all read only line 0, which set() has already prefetched. The lookup
 * prefetches the tail of the chosen element; it is first needed on the second
 * bit of the chunk.
 *
 * Ranking: the elements of a group are ranked from most recently used (rank 0)
 * to least recently used. Only the ranking bytes are reordered; the elements
 * themselves never move. An element keeps its index, checksum, root and tail
 * for as long as it lives. Moving an element to the front is a few shifts on
 * one 64 bit word and copies no counters, so the pointers that ActiveContext
 * keep into the group stay valid for the whole chunk.
 *
 * Lookup: the group is searched in rank order for the context's checksum. If
 * found, that element is moved to the front. If not found, an empty element is
 * used; if there is none, the least used of the lower-ranked elements is
 * replaced. The Protected top-ranked elements are never replaced. This way the
 * element claimed last is always safe, and two contexts that alternate in the
 * same group cannot keep evicting each other. No extra flag is needed for
 * this: the rank alone tells whether an element is protected.
 *
 * Protection also limits the harm when several contexts land in the same group on
 * the same chunk. Right after a context's lookup its element is at rank 0, so with
 * Protected = k it cannot be replaced until k more contexts have looked up the
 * same group. If even more contexts collide, the element may be replaced, and two
 * contexts then share it until the end of the chunk.
 *
 * A replaced element is cleared first, so it only ever holds the statistics of
 * one context: nothing leaks from the contexts that used it before.
 *
 * New elements always go to the front, so in every group all used elements are
 * before all empty ones. The search can therefore stop at the first
 * empty element.
 */
class StationaryMapForJpegModel : IPredictor
{
public:
  static constexpr int MIXERINPUTS = 3; // per context

private:
  static constexpr uint32_t InputBits = 3; // it's specifically designed for the JpegModel -> it's fixed + the memory layout also depends on it, 
  static constexpr uint32_t Counters = (1u << InputBits) - 1; // counters in one bit tree (one element)
  static constexpr uint32_t ElementsPerGroup = 8; // ranked from most to least recently used
  static constexpr uint32_t Protected = 1; // this many top-ranked elements are never replaced
  static constexpr int ChecksumBits = 24;

  static_assert(ElementsPerGroup == 8, "the ranking stores one byte per rank in a 64 bit word");
  static_assert(Protected >= 1 && Protected < ElementsPerGroup, "at least one element must be protected and at least one replaceable");
  static_assert(ChecksumBits == 24, "chkLo + chkHi hold exactly 24 bits");

  /**
   * The starting ranking: element 0 at rank 0, element 1 at rank 1, and so on.
   *
   * A ranking is 8 bytes, one per rank: byte r (counting from the lowest byte)
   * holds the index of the element at rank r. In Identity every byte holds its
   * own position, so reading it byte by byte from the lowest gives 0, 1, ..., 7:
   *
   *   Identity = 0x 07 06 05 04 03 02 01 00
   *   rank:         7  6  5  4  3  2  1  0
   *
   * Every valid ranking is a reordering of these 8 values: each element index
   * appears at exactly one rank.
   *
   * Group::order stores the ranking XORed with Identity, not the ranking itself:
   *
   *   stored  = ranking ^ Identity   (when writing)
   *   ranking = stored  ^ Identity   (when reading; XOR undoes itself)
   *
   * The reason is reset(): it zeroes the whole table with a single memset. For
   * a zeroed group, reading gives 0 ^ Identity = Identity, which is a valid
   * starting ranking. Every element is empty (its chkLo is 0), so the order of
   * the elements does not matter yet, as long as each index appears once.
   *
   * If the ranking were stored directly, a zeroed group would read as
   * 0x0000000000000000: element 0 at every rank. Elements 1 to 7 would then
   * never be found by the search or chosen for replacement, and the group would
   * behave as if it had only one element. Nothing would fail or crash; the
   * model would just silently get worse.
   */
  static constexpr uint64_t Identity = UINT64_C(0x0706050403020100);

  struct alignas(64) Group
  {
    // First cache line: everything the lookup needs.
    uint16_t chkLo[ElementsPerGroup]; // low 16 bits of the checksum; 0 means the element is empty
    uint32_t root[ElementsPerGroup]; // counters[0] of each element: n0 in the high half, n1 in the low half
    uint64_t order; // the ranking XOR Identity; byte r of the ranking is the index of the element at rank r
    uint8_t chkHi[ElementsPerGroup]; // high 8 bits of the checksum; only read when chkLo matches
    // Remaining cache lines: only the tail of the element in use is touched.
    uint32_t tail[ElementsPerGroup][Counters - 1]; // counters[1..] of each element
  };

  static_assert(sizeof(Group) == 256, "unexpected padding in Group");

  /** The state of one context during the current chunk. */
  struct ActiveContext
  {
    Group* group; // the group of the current context
    uint32_t* counter; // the counter of the current bit: set by mix(), updated by update()
    uint32_t* tail; // counters[1] of the element in use; valid after the first mix() of the chunk
    uint32_t chk; // the 24 bit checksum of the current context; its low 16 bits are never 0
  };

  const Shared* const shared;
  Array<Group, 64> data;
  Array<ActiveContext> activeContexts;
  Random rnd;
  const uint32_t numContexts;
  const int groupBits; // log2 of the number of groups
  int scale;
  uint32_t bCount; // bits modelled so far in this chunk; used by asserts only
  uint32_t b; // the node of the bit tree for the current bit
  bool subscribed;

  /** Returns the index of the element at rank r. */
  static ALWAYS_INLINE uint32_t elementAt(const uint64_t ranking, const uint32_t r) {
    return static_cast<uint32_t>(ranking >> (8 * r)) & 0xff;
  }

  /** Returns the ranking with rank r moved to the front; ranks 0..r-1 each move back by one. */
  static ALWAYS_INLINE uint64_t moveToFront(const uint64_t ranking, const uint32_t r) {
    const uint64_t mask = ~UINT64_C(0) >> (8 * (ElementsPerGroup - 1 - r)); // the bytes of ranks 0..r
    return (ranking & ~mask) | ((ranking << 8) & mask) | ((ranking >> (8 * r)) & 0xff);
  }

  /** (Roughly) how many chunks an element has been used for. Elements with fewer uses are replaced first. */
  static ALWAYS_INLINE uint32_t useCount(const uint32_t root) {
    return (root >> 16) + (root & 0xffff); // the root counter is updated exactly once per chunk
  }

  /** Clears element e of group g and assigns it to checksum chk. */
  static void claim(Group* g, uint32_t e, uint32_t chk);

  /** Chooses the rank of the element to replace when the group is full and the context was not found. */
  uint32_t victimRank(const Group* g, uint64_t ranking);

  /** Finds (or claims) the element of ctx's context and points ctx.counter and ctx.tail at it. */
  void lookup(ActiveContext& ctx);

public:
  /**
    * Uses 2^bitsOfMemory bytes of memory in total, shared by all contexts.
    * @param numContexts the number of contexts in this bank
    * @param bitsOfMemory log2 of the table size in bytes
    * @param inputBits must equal InputBits; call set() once every inputBits bits
    * @param scale is input scaling, 64 = neutral
    */
  StationaryMapForJpegModel(const Shared* sh, uint32_t numContexts, int bitsOfMemory, int inputBits, int scale = 64);

  /**
    * Starts a new chunk of InputBits bits for context i and prefetches its group.
    * ctxHash must be a hash, not a raw context value: pass the result of hash() directly.
    */
  void set(uint32_t i, uint64_t ctxHash);

  /**
    * Adds MIXERINPUTS inputs to m for context i and returns a scaled stretched probability.
    */
  int mix(Mixer& m, uint32_t i);

  void setScale(int scale);
  void reset();
  void update() override;
};
