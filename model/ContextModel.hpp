#ifndef PAQ8PX_CONTEXTMODEL_HPP
#define PAQ8PX_CONTEXTMODEL_HPP

#include "../BlockType.hpp"
#include "../Mixer.hpp"
#include "../MixerFactory.hpp"
#include "../Models.hpp"
#include "ContextModelGeneric.cpp"

/**
 * This combines all the context models with a Mixer.
 */
class ContextModel {
    Shared * const shared;
    Models * const models;
    BlockType nextBlockType = BlockType::DEFAULT;
    BlockType blockType = BlockType::DEFAULT;
    int blockSize = 1;
    int blockInfo = 0;
    int bytesRead = 0;
    bool readSize = false;

    ContextModelGeneric contextModelGeneric{ shared, models }; // we always need a generic model, so we declare and instatiate it here
    IContextModel* selectedContextModel = &contextModelGeneric;

public:
    ContextModel(Shared* const sh, Models* const models);
    int p();
};

#endif //PAQ8PX_CONTEXTMODEL_HPP
