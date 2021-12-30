#pragma once

#include "AdaptiveMap.hpp"

/**
 * An APM maps a probability and a context to a new probability.
 */
class APM : public AdaptiveMap, IPredictor {
private:
    const int N; /**< Number of contexts */
    const int steps;
    int cxt; /**< context index of last prediction */
public:
  int limit; //1..1023
  /**
     * Creates with @ref n contexts using 4*s*n bytes memory.
     * @param n the number of contexts
     * @param s the number of steps
     */
    APM(const Shared* const sh, int n, int s, int limit);
    ~APM() override = default;

    /**
     * a.update() updates probability map. y=(0..1) is the last bit
     */
    void update() override;

    /**
     * Returns a new probability (0..4095) like with @ref StateMap.
     * cx=(0..n-1) is the context.
     * pr=(0..4095) is considered part of the context.
     * The output is computed by interpolating @ref pr into @ref steps ranges nonlinearly
     * with smaller ranges near the ends. The initial output is pr.
     * lim=(0..1023): set a lower limit (like 255) for faster adaptation.
     * @param pr the initial probability. Considered part of the context.
     * @param cx the context
     * @param lim limit; lower limit means faster adaptation
     * @return the adjusted probability
     */
    int p(int pr, int cx);
};
