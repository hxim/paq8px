#pragma once

#include "../Array.hpp"
#include "../Mixer.hpp"
#include "../Random.hpp"
#include "../Shared.hpp"
#include "../StateMap.hpp"
#include "../StateTable.hpp"
#include "../Stretch.hpp"
#include "../Utils.hpp"
#include "DmcNode.hpp"
#include <cassert>
#include <cstdint>

/**
 * Model using DMC (Dynamic Markov Compression).
 *
 * The bitwise context is represented by a state graph.
 *
 * See the original paper: http://webhome.cs.uvic.ca/~nigelh/Publications/DMC.pdf
 * See the original DMC implementation: http://maveric0.uwaterloo.ca/ftp/dmc/
 *
 * Main differences:
 * - Instead of floats we use fixed point arithmetic.
 * - The threshold for cloning a state increases gradually as memory is used up.
 * - For probability estimation each state maintains both a 0,1 count ("c0" and "c1")
 *   and a bit history ("state"). The 0,1 counts are updated adaptively favoring newer events.
 *   The bit history state is mapped to a probability adaptively using a @ref StateMap.
 * - The predictions of multiple "DmcModel"s are combined and stabilized in a @ref DmcForest.
 *
 * Max memory use:
 * 12 bytes (DMC node size, fixed) * 2^28 nodes (that is 268'435'456, max) = 3'221'225'472 (~ 3 GB)
 *
 */
class DmcModel : protected IPredictor {
private:
    constexpr static uint64_t dmcNodesBase = (255 * 256); /**< 65280 */
    constexpr static uint64_t dmcNodesMax = (UINT64_C(1) << 28); /**< 268'435'456 */
    const Shared * const shared;
    Random rnd;
    Array<DMCNode> t; /**< state graph */
    StateMap sm; /**< stateMap for bit history states */
    uint32_t top {}; /**< index of first unallocated node (i.e. number of allocated nodes) */
    uint32_t curr {}; /**< index of current node */
    uint32_t threshold {}; /**< cloning threshold parameter: fixed point number like c0,c1 */
    uint32_t thresholdFine {}; /**< "threshold" scaled by 11 bits used for increasing the threshold in finer steps */
    uint32_t extra {}; /**< this value is used for approximating state graph maturity level when the state graph is already full */

    /**
     * Helper function: adaptively increment a counter.
     * @param x a fixed point number as c0, c1.
     * @param increment either 0 or 1.
     * @return
     */
    [[nodiscard]] static auto incrementCounter(uint32_t x, uint32_t increment) -> uint32_t;

public:
    DmcModel(const Shared* const sh, uint64_t dmcNodes, uint32_t thStart);

    /**
     * Initialize the state graph to a bytewise order 1 model.
     * See an explanation of the initial structure in:
     * http://wing.comp.nus.edu.sg/~junping/docs/njp-icita2005.pdf
     * @param thStart
     */
    void resetStateGraph(uint32_t thStart);

    /**
     * Update state graph.
     */
    void update();

    /**
     * Determine if the state graph is full or not.
     * @return
     */
    [[nodiscard]] auto isFull() const -> bool;

    /**
     * Stretched probability based on counts.
     * @return
     */
    [[nodiscard]] auto st1() const -> int;

    /**
     * Stretched probability based on state.
     * @return
     */
    [[nodiscard]] auto st2() -> int;

    /**
    * Weighted stretched probability.
    * @return
    */
    [[nodiscard]] auto stw() -> int;
};
