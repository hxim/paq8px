#ifndef PAQ8PX_NORMALMODEL_HPP
#define PAQ8PX_NORMALMODEL_HPP

#include "../ContextMap2.hpp"

/**
 * Model for order 0-14 contexts
 * Contexts are hashes of previous 0..14 bytes.
 * Order 0..6, 8 and 14 are used for prediction.
 * Note: order 7+ contexts are modeled by matchModel as well.
 */
class NormalModel {
private:
    static constexpr int nCM = 9;
    static constexpr int nSM = 3;
    Shared * const shared;
    ContextMap2 cm;
    StateMap smOrder0Slow;
    StateMap smOrder1Slow;
    StateMap smOrder1Fast;
public:
    static constexpr int MIXERINPUTS =
      nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY) + 
      nSM; //66
    static constexpr int MIXERCONTEXTS_PRE = 64;
    static constexpr int MIXERCONTEXTS_POST = 1024 + 256 + 512 + 256 + 256 + 1536; //3840
    static constexpr int MIXERCONTEXTSETS_PRE = 1;
    static constexpr int MIXERCONTEXTSETS_POST = 6;
    NormalModel(Shared* const sh, const uint64_t cmSize);
    void reset();

    /**
     * update order 1..14 context hashes.
     * Note: order 0 context does not need an update so its hash never changes.
     */
    void updateHashes();
    void mix(Mixer &m);

    /**
     * setting more mixer contexts in the generic case (i.e. not using the special models such as image, audio, jpeg models)
     * @param m
     */
    void mixPost(Mixer &m);
};

#endif //PAQ8PX_NORMALMODEL_HPP
