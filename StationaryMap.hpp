#pragma once

#include "IPredictor.hpp"
#include "Hash.hpp"
#include "Mixer.hpp"
#include "Stretch.hpp"
#include "UpdateBroadcaster.hpp"

/**
 * Map for modelling contexts of (nearly-)stationary data.
 * The context is looked up directly. For each bit modelled, the exact counts of 0s and 1s are stored.
 * 
 */
class StationaryMap : IPredictor {
public:
    static constexpr int MIXERINPUTS = 3;

private:
    const Shared * const shared;
    Array<uint32_t> data;
    const uint32_t mask, stride, bTotal;
    int scale, rate;
    uint32_t context;
    uint32_t bCount;
    uint32_t b;
    uint32_t *cp;

public:
    /**
     * Construct using (2^(BitsOfContext+2))*((2^InputBits)-1) bytes of memory.
     * @param bitsOfContext How many bits to use for each context. Higher bits are discarded.
     * @param inputBits How many bits [1..8] of input are to be modelled for each context. New contexts must be set at those intervals.
     * @param scale
     * @param rate Use 16 for near-stationary modelling (default), smaller values may be used for tuning adaptivity
     */
    StationaryMap(const Shared* const sh, const int bitsOfContext, const int inputBits, const int scale = 64, const int rate = 16);

    /**
     * ctx must be a direct context (no hash)
     * @param ctx
     */
    void set(uint32_t ctx);
    void setScale(int scale);
    void setRate(int rate);
    void reset();
    void update() override;
    void mix(Mixer &m);
};
