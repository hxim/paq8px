#pragma once

#include "../ContextMap2.hpp"
#include "../Shared.hpp"
#include <cstdint>

class ChartModel {
private:
  static constexpr int nCM = (3 * 4) + 3 + (8 + 8 + 8) * 3; //87
  static constexpr int nCM_TEXT = (3 * 4) + 2 + (8) * 3; //38
    const Shared * const shared;
    ContextMap2 cm;
    uint32_t chart[3*8]{};
    uint32_t charGroup{};
    uint8_t indirect1[1024+64]{};
    uint8_t indirect2[256]{};
    uint8_t indirect3[256 * 256]{};
public:
    static constexpr int MIXERINPUTS = nCM * (ContextMap2::MIXERINPUTS); // 348
    static constexpr int MIXERINPUTS_TEXT = nCM_TEXT * (ContextMap2::MIXERINPUTS); // 152
    static constexpr int MIXERCONTEXTS = 0;
    static constexpr int MIXERCONTEXTSETS = 0;
    explicit ChartModel(const Shared* const sh, const uint64_t size);
    void mix(Mixer &m);
};
