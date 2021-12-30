#pragma once

// TODO(epsteina): update this documentation
/**
context map for large contexts.
maps to a bit history state, a 3 mostRecentlyUsed byte history, and 1 byte RunStats.

Bit and byte histories are stored in a hash table with 64 byte buckets.
The buckets are indexed by a context ending after 0, 2 or 5 bits of the
current byte. Thus, each byte modeled results in 3 main memory accesses
per context, with all other accesses to cache.

On a byte boundary (bit 0), only 3 of the 7 bit history states are used.
Of the remaining 4 bytes, 1 byte is used as a run length (the consecutive
occurrences of the previously seen byte), 3 are used to store the last
3 distinct bytes seen in this context. The byte history is then combined
with the bit history states to provide additional states that are then
mapped to predictions.
*/

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
#include "UpdateBroadcaster.hpp"

#define CM_USE_NONE 0
#define CM_USE_RUN_STATS 1
#define CM_USE_BYTE_HISTORY 2
#define CM_DEFERRED 64
#define CM_SKIPPED_CONTEXT 128

class ContextMap2 : IPredictor {
public:
    static constexpr int MIXERINPUTS = 4;
    static constexpr int MIXERINPUTS_RUN_STATS = 1;
    static constexpr int MIXERINPUTS_BYTE_HISTORY = 2;

private:

  struct ContextInfo {
    HashElementForContextMap* slot0; /**< pointer to current byte history in slot0 */
    HashElementForContextMap* slot012; /**< pointer to current bit history states in current slot (either slot0 or slot1 or slot2) */
    uint32_t tableIndex; /**< @ref C whole byte context hashes */
    uint16_t tableChecksum; /**< @ref C whole byte context checksums */
    uint8_t flags;
    HashElementForContextMap bitStateTmp;
  };

    const Shared * const shared;
    Random rnd;
    const uint32_t C; /**< max number of contexts */
    Array<Bucket16<HashElementForContextMap, 7>> hashTable; /**< bit and byte histories (statistics) */
    Array<ContextInfo> contextInfoList;
    StateMap runMap;
    StateMap stateMap;
    StateMap bhMap8B;
    StateMap bhMap12B;
    uint32_t index; /**< next context to set by @ref ContextMap2::set(), resets to zero after every round */
    const uint32_t mask;
    const int hashBits;
    int scale;
    uint8_t contextflagsAll;

    void updatePendingContextsInSlot(HashElementForContextMap* const p, uint32_t c);
    void updatePendingContexts(uint32_t ctx, uint16_t checksum, uint32_t c);
    size_t getStateByteLocation(const uint32_t bpos, const uint32_t c0);

public:
    int order = 0; // is set after mix()
    /**
     * Construct using @ref size bytes of memory for @ref contexts contexts.
     * @param size bytes of memory to use
     * @param contexts max number of contexts
     * @param scale
     * @param uw
     */
    ContextMap2(const Shared* const sh, uint64_t size, uint32_t contexts, int scale);

    /**
     * Set next whole byte context to @ref ctx.
     * @param ctx
     */
    void set(uint8_t ctxflags, const uint64_t ctx);
    void skip(const uint8_t ctxflags);
    void skipn(const uint8_t ctxflags, const int n);
    void update() override;
    void setScale(const int Scale);
    void mix(Mixer &m);
};
