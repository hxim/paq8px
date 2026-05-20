#pragma once

#include "DivisionTable.hpp"
#include "AdaptiveMap.hpp"
#include "StateTable.hpp"
#include "UpdateBroadcaster.hpp"
#include "IPredictor.hpp"

enum class StateMapType {
  Generic, BitHistory, Run
};

/**
 * A @ref StateMap maps a context to a probability.
 */
class StateMap : public AdaptiveMap, protected IPredictor {
protected:
  const uint32_t numContextSets; /**< Number of context sets */
  const uint32_t numContextsPerSet; /**< Number of contexts in each context set */
  uint32_t currentContextSetIndex; /**< Number of context indexes present in cxt array (0..numContextSets-1) */
  Array<uint32_t> cxt; /**< context index of last prediction per context set */
public:
  int limit; //1..1023
 

  /**
    * Creates a @ref StateMap with @ref n contexts using 4*n bytes memory.
    * @param s
    * @param n number of contexts
    * @param lim
    * @param mapType
    */
  StateMap (const Shared* const sh, int s, int n, int lim, StateMapType mapType);
  ~StateMap() override = default;

  void update() override;

  /**
    * Call @ref p1() when there is only 1 context set.
    * No need to call @ref subscribe().
    * @param cx
    * @return
    */
  int p1(uint32_t cx);

  /**
    * Call @ref p2() for each context when there are more context sets, finally call @ref subscribe() once.
    * sm.p(y, cx, limit) converts state @ref cx (0..n-1) to a probability (0..4095)
    * that the next y=1, updating the previous prediction with y (0..1).
    * limit (1..1023, default 1023) is the maximum count for computing a
    * prediction.  Larger values are better for stationary sources.
    * @param s
    * @param cx
    * @return
    */
  int p2(uint32_t s, uint32_t cx);
  void subscribe();

  /**
    * Call @ref skip() instead of @ref p2() when the context is unknown or uninteresting.
    * @remark no need to call @ref skip() when there is only 1 context or all contexts will be skipped.
    * @param s
    */
  void skip(uint32_t s);
  void print() const;
};
