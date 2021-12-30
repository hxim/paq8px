#pragma once

#include "IPredictor.hpp"
#include "Bucket16.hpp"
#include "Hash.hpp"
#include "Mixer.hpp"
#include "Stretch.hpp"
#include "UpdateBroadcaster.hpp"

/**
 * Map for modelling contexts of (nearly-)stationary data.
 * The context is a hash. For each bit modelled, the exact counts of 0s and 1s are stored.
 *
 */

class LargeStationaryMap : IPredictor {
public:
    static constexpr int MIXERINPUTS = 3;

private:
    const Shared * const shared;
    Random rnd;
    Array<Bucket16<HashElementForStationaryMap, 7>> data;
    const uint32_t hashBits;
    int scale, rate;

    const uint32_t numContexts; /**< Maximum supported contexts */
    uint32_t currentContextIndex; /**< Number of context indexes present in cxt array (0..numContexts-1) */
    Array<uint64_t> contextHashes; /**< context index of last prediction per context */

public:
    /**
     * Construct using 2^hashBits * sizeof(Bucket16) bytes of memory for storing a maximum of 2^hashBits * ElementsInBucket
     * That is:
      *   hashBits    |       memory          |   maximum number of contexts
      *       10      |  2^10 * 42 =   43 KB  |         7 K  (~12 bits)
      *       11      |  2^11 * 42 =   86 KB  |        14 K  (~13 bits)
      *       12      |  2^12 * 42 =  172 KB  |        28 K  (~14 bits)
      *       13      |  2^13 * 42 =  344 KB  |        57 K  (~15 bits)
      *       14      |  2^14 * 42 =  688 KB  |       114 K  (~16 bits)
      *       15      |  2^15 * 42 =  1.3 MB  |       229 K  (~17 bits)
      *       16      |  2^16 * 42 =  2.7 MB  |       458 K  (~18 bits)
      *       17      |  2^17 * 42 = 11.0 MB  |       1.8 M  (~19 bits)
      *       18      |  2^18 * 42 = 22.0 MB  |       3.6 M  (~20 bits)
      *       19      |  2^19 * 42 = 44.0 MB  |       7.3 M  (~21 bits)
      *       20      |  2^20 * 42 = 88.0 MB  |      14.6 M  (~22 bits)
      *      ...              ...                         ...
      * 
     * @param scale
     * @param rate use 16 near-stationary modelling (default), smaller values may be used for tuning adaptivity
     */
    LargeStationaryMap(const Shared* const sh, const int contexts, const int hashBits, const int scale = 64, const int rate = 16);

    /**
     * ctx must be a hash
     * @param ctx
     */
    void set(const uint64_t contextHash);
    void setscale(const int scale);
    void reset();
    void update() override;
    void update(uint32_t *cp);
    void mix(Mixer &m);
    void subscribe();
    void skip();

};
