#pragma once

#include "../MTFList.hpp"
#include "../Hash.hpp"
#include "../IndirectContext.hpp"
#include "../Mixer.hpp"
#include "../Shared.hpp"
#include "../LargeStationaryMap.hpp"
#include "../StationaryMap.hpp"

class SparseMatchModel {
private:
    static constexpr int numHashes = 4;
    static constexpr int nLSM = 1;
    static constexpr int nSM = 3;
    const Shared * const shared;
    enum Parameters : uint32_t {
        MaxLen = 0xFFFF, // longest allowed match
        MinLen = 3, // default minimum required match length
    };
    struct SparseConfig {
        uint32_t offset = 0; // number of last input bytes to ignore when searching for a match
        uint32_t stride = 1; // look for a match only every stride bytes after the offset
        uint32_t deletions = 0; // when a match is found, ignore these many initial post-match bytes, to model deletions
        uint32_t minLen = MinLen;
        uint8_t bitMask = 0xFF; // match every byte according to this bit mask
    };
    const SparseConfig sparse[numHashes] = {{0, 1, 0, 5, 0xDF},
                                            {1, 1, 0, 4, 0xFF},
                                            {0, 2, 0, 4, 0xDF},
                                            {0, 1, 0, 5, 0x0F}};
    Array<uint32_t> table;
    LargeStationaryMap mapL;
    StationaryMap maps[nSM];
    IndirectContext<uint8_t> iCtx8 {19, 1}; // BitsPerContext, InputBits
    IndirectContext<uint16_t> iCtx16 {16, 8}; // BitsPerContext, InputBits
    MTFList list {numHashes};
    uint32_t hashes[numHashes] {};
    uint32_t hashIndex = 0; // index of hash used to find current match
    uint32_t length = 0; // rebased length of match (length=1 represents the smallest accepted match length), or 0 if no match
    uint32_t index = 0; // points to next byte of match in buf, 0 when there is no match
    uint8_t expectedByte = 0; // prediction is based on this byte (buf[index]), valid only when length>0
    bool valid = false;
    const uint32_t mask;
    const int hashBits;

public:
    static constexpr int MIXERINPUTS = 
      3 + 
      nLSM * LargeStationaryMap::MIXERINPUTS + 
      nSM * StationaryMap::MIXERINPUTS; // 15
    static constexpr int MIXERCONTEXTS = numHashes * (64 + 2048); // 8448
    static constexpr int MIXERCONTEXTSETS = 2;
    explicit SparseMatchModel(const Shared* const sh, uint64_t size);
    void update();
    void mix(Mixer &m);
};
