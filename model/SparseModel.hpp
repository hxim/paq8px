#ifndef PAQ8PX_SPARSEMODEL_HPP
#define PAQ8PX_SPARSEMODEL_HPP

#include "../ContextMap2.hpp"
#include "../Mixer.hpp"
#include "../Shared.hpp"
#include <cassert>
#include <cstdint>

/**
 * Model order 1-2-3 contexts with gaps.
 */
class SparseModel {
private:
    static constexpr int nCM = 28 + 3;
    const Shared * const shared;
    ContextMap2 cm;
    uint32_t ctx = 0;
public:
    static constexpr int MIXERINPUTS = nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 190
    static constexpr int MIXERCONTEXTS = 4 * 256;
    static constexpr int MIXERCONTEXTSETS = 1;
    explicit SparseModel(const Shared* const sh, uint64_t size);
    void mix(Mixer &m);
};

#endif //PAQ8PX_SPARSEMODEL_HPP
