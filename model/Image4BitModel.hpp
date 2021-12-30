#pragma once

#include "../IPredictor.hpp"
#include "../Shared.hpp"
#include "../Hash.hpp"
#include "../Ilog.hpp"
#include "../Mixer.hpp"
#include "../Random.hpp"
#include "../RingBuffer.hpp"
#include "../StateMap.hpp"
#include "../LargeStationaryMap.hpp"
#include "../HashElementForBitHistoryState.hpp"

/**
 * Model for 4-bit image data
 */
class Image4BitModel : IPredictor {
private:
    static constexpr int S = 14; /**< number of contexts */
    const Shared * const shared;
    Random rnd;
    Array<Bucket16<HashElementForBitHistoryState, 7>> hashTable;
    const uint32_t mask;
    const int hashBits;
    Array<uint64_t> hashes;
    StateMap sm;
    LargeStationaryMap mapL;
    uint8_t *cp[S] {}; /**< context pointers  */
    uint8_t WW = 0, W = 0, NWW = 0, NW = 0, N = 0, NE = 0, NEE = 0, NNWW = 0, NNW = 0, NN = 0, NNE = 0, NNEE = 0;
    int w = 0, col = 0, line = 0, run = 0, prevColor = 0;

public:
  static constexpr int MIXERINPUTS =
      1 + // individual m.add
      S * 3 + 
      S * LargeStationaryMap::MIXERINPUTS;
    static constexpr int MIXERCONTEXTS = 256 + 512 + 512 + 1024 + 16 + 1; /**< 2321 */
    static constexpr int MIXERCONTEXTSETS = 6;
    explicit Image4BitModel(const Shared* const sh, uint64_t size);
    void setParam(int widthInBytes);
    void update() override;
    void mix(Mixer &m);
};
