#pragma once

#include "UpdateBroadcaster.hpp"
#include "Mixer.hpp"

/**
 * For training @ref NormalModel, @ref WordModel and @erf ExeModel
 */
class DummyMixer : public Mixer {
public:
    DummyMixer(const Shared* const sh, int n, int m, int s);
    void update() override;
    auto p() -> int override;

    void setScaleFactor(const int /*sf0*/, const int /*sf1*/) override {}
    void promote(int) override {}
};
