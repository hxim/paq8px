#ifndef PAQ8PX_SPARSEBITMODEL_HPP
#define PAQ8PX_SPARSEBITMODEL_HPP

#include "../ContextMap2.hpp"
#include "../Shared.hpp"
#include <cstdint>

/**
 * Modeling bit contexts with various masks.
 */
class SparseBitModel {
private:
    static constexpr int nCM = 10;
    const Shared * const shared;
    ContextMap2 cm;
public:
    static constexpr int MIXERINPUTS = nCM * (ContextMap2::MIXERINPUTS); // 40
    static constexpr int MIXERCONTEXTS = 0;
    static constexpr int MIXERCONTEXTSETS = 0;
    explicit SparseBitModel(const Shared* const sh, uint64_t size);
    void mix(Mixer &m);
};

#endif //PAQ8PX_SPARSEBITMODEL_HPP
