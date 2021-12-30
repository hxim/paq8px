#pragma once

#include "../Shared.hpp"
#include "../ContextMap2.hpp"

class NestModel {
private:
    static constexpr int nCM = 12;
    const Shared * const shared;
    int ic = 0, bc = 0, pc = 0, vc = 0, qc = 0, lvc = 0, wc = 0, ac = 0, ec = 0, uc = 0, sense1 = 0, sense2 = 0, w = 0;
    ContextMap2 cm;

public:
    static constexpr int MIXERINPUTS = nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS); // 60
    static constexpr int MIXERCONTEXTS = 0;
    static constexpr int MIXERCONTEXTSETS = 0;
    explicit NestModel(const Shared* const sh, uint64_t size);
    void mix(Mixer &m);
};
