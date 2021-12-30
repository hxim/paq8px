#pragma once

#include "../BlockType.hpp"
#include "../Mixer.hpp"
#include "../MixerFactory.hpp"
#include "../Models.hpp"

/**
 * This combines all the context models with a Mixer.
 */
class ContextModel {
    Shared * const shared;
    Models * const models;
    const MixerFactory* const mixerFactory;

    IContextModel* selectedContextModel = nullptr;

public:
  ContextModel(Shared* const sh, Models* const models, const MixerFactory* const mixerFactory);
    int p();
};
