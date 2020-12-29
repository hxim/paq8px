#ifndef PAQ8PX_CONTEXTMAP2_HPP
#define PAQ8PX_CONTEXTMAP2_HPP

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
#define CM_SKIPPED_CONTEXT 128

class ContextMap2 : IPredictor {
public:
    static constexpr int MIXERINPUTS = 4;
    static constexpr int MIXERINPUTS_RUN_STATS = 1;
    static constexpr int MIXERINPUTS_BYTE_HISTORY = 2;

private:
    const Shared * const shared;
    Random rnd;
    const uint32_t C; /**< max number of contexts */
    Array<Bucket16<HashElementForContextMap, 7>> table; /**< bit and byte histories (statistics) */
    Array<uint8_t *> bitState; /**< @ref C pointers to current bit history states */
    Array<uint8_t *> bitState0; /**< First element of 7 element array containing bitState[i] */
    Array<uint8_t *> byteHistory; /**< @ref C pointers to run stats plus byte history, 4 bytes, [RunStats,1..3] */
    Array<uint32_t> contexts; /**< @ref C whole byte context hashes */
    Array<uint16_t> checksums; /**< @ref C whole byte context checksums */
    Array<uint8_t> contextflags;
    StateMap runMap;
    StateMap stateMap;
    StateMap bhMap8B;
    StateMap bhMap12B;
    uint32_t index; /**< next context to set by @ref ContextMap2::set(), resets to zero after every round */
    const uint32_t mask;
    const int hashBits;
    int scale;
    uint8_t contextflagsAll;

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
    void set(const uint8_t ctxflags, uint64_t ctx);
    void skip(const uint8_t ctxflags);
    void skipn(const uint8_t ctxflags, int n);
    void update() override;
    void setScale(int Scale);
    void mix(Mixer &m);
};

#endif //PAQ8PX_CONTEXTMAP2_HPP
