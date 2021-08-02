#ifndef PAQ8PX_CHARTMODEL_HPP
#define PAQ8PX_CHARTMODEL_HPP

#include "../ContextMap2.hpp"
#include "../Shared.hpp"
#include <cstdint>

class ChartModel {
private:
    static constexpr int nCM1 = (2*8+4)*3;
    static constexpr int nCM2 = 3*5+8;
    const Shared * const shared;
    ContextMap2 cm, cn;
    uint32_t chart[2*8+4]{};
    uint8_t indirect1[1024+64]{};
    uint8_t indirect2[256]{};
    uint8_t indirect3[256 * 256]{};
public:
    static constexpr int MIXERINPUTS = (nCM1+nCM2) * (ContextMap2::MIXERINPUTS); // 380
    static constexpr int MIXERCONTEXTS = 0;
    static constexpr int MIXERCONTEXTSETS = 0;
    explicit ChartModel(const Shared* const sh, const uint64_t size1, const uint64_t size2);
    void mix(Mixer &m);
};

#endif //PAQ8PX_CHARTMODEL_HPP
