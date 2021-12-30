#pragma once

#include "../Shared.hpp"
#include "../Array.hpp"
#include "DmcModel.hpp"

/**
 * This class solves two problems of the DMC model.
 * 1) The DMC model is a memory hungry algorithm. In theory it works best when it can clone
 *    nodes forever. But when the state graph is full you can't clone nodes anymore.
 *    You can either i) reset the model (the state graph) and start over
 *    or ii) you can keep updating the counts forever in the already fixed state graph. Both
 *    choices are troublesome: i) resetting the model degrades the predictive power significantly
 *    until the graph becomes large enough again and ii) a fixed structure can't adapt anymore.
 *    To solve this issue:
 *    Ten models with different arguments work in tandem. Only eight of the ten models
 *    are reset periodically. Due to their different cloning threshold arguments and
 *    different state graph sizes they are reset at different points in time.
 *    The remaining two models (having the highest threshold and largest state graph) are
 *    never reset and are beneficial for semi-stationary files.
 * 2) The DMC model is sensitive to the cloning threshold parameter. Some files prefer
 *    a smaller threshold other files prefer a larger threshold.
 *    The difference in terms of compression is significant.
 *    To solve this issue DMC models with different thresholds are used and their
 *    predictions are combined.
 *
 *    Disadvantages: with the same memory requirements we have less number of nodes
 *    in each model. Also keeping more models updated at all times requires more
 *    calculations and more memory access than updating one model only.
 *    Advantage: more stable and better compression - even with reduced number of nodes.
 *
 * Further notes:
 *    Extremely small initial threshold arguments (i) help the state graph become large faster
 *    and model longer input bit sequences sooner. Moreover (ii) when using a small threshold
 *    parameter the split counts c0 and c1 will be small after cloning, and after updating them
 *    with 0 and 1 the prediction p=c1/(c0+c1) will be biased towards these latest events.
 */

class DmcForest {
private:
    static constexpr uint32_t MODELS = 10; /**< 8 fast and 2 slow models */
    static constexpr uint32_t dmcParams[MODELS] = {2, 32, 64, 4, 128, 8, 256, 16, 1024, 1536};
    static constexpr uint32_t dmcMem[MODELS] = {6, 10, 11, 7, 12, 8, 13, 9, 2, 2};
    const Shared * const shared;
    Array<DmcModel *> dmcModels;

public:
    static constexpr int MIXERINPUTS = 2 + 8 / 2; /**< 6 : fast models (2 individually) + slow models (8 combined pairwise) */
    static constexpr int MIXERCONTEXTS = 0;
    static constexpr int MIXERCONTEXTSETS = 0;
    explicit DmcForest(const Shared* const sh, uint64_t size);
    ~DmcForest();

    /**
     * Update and predict all the DMC models.
     * @param m
     */
    void mix(Mixer &m);
};
