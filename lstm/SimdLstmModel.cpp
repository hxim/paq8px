#include "../Stretch.hpp"
#include "SimdLstmModel.hpp"
#include <cstring>
#include <algorithm>

SIMDLstmModel::SIMDLstmModel(
    const Shared* const sh,
    SIMDType simdType,
    size_t const num_cells,
    size_t const num_layers,
    size_t const horizon,
    float const learning_rate,
    float const gradient_clip)
    : LstmModel(sh)
    , simd(simdType)
    , shape{ 0, this->alphabetSize, num_cells, num_layers, horizon }
    , lstm(simdType, shape, learning_rate, gradient_clip)
    , modelType(LSTM::Model::Type::Default)
    , pModelType(LSTM::Model::Type::Default)
    , pBlockType(BlockType::Count)
{
    if (this->shared->GetOptionTrainLSTM()) {
        repo[LSTM::Model::Type::Default] = std::unique_ptr<LSTM::Model>(new LSTM::Model(shape));
        lstm.SaveModel(*repo[LSTM::Model::Type::Default]);
        
        repo[LSTM::Model::Type::English] = std::unique_ptr<LSTM::Model>(new LSTM::Model(shape));
        repo[LSTM::Model::Type::English]->LoadFromDisk("english.rnn", 4, 1);
        
        repo[LSTM::Model::Type::x86_64] = std::unique_ptr<LSTM::Model>(new LSTM::Model(shape));
        repo[LSTM::Model::Type::x86_64]->LoadFromDisk("x86_64.rnn", 4, 1);
    }
}

void SIMDLstmModel::mix(Mixer& m) {
    uint8_t const bpos = this->shared->State.bitPosition;
    uint8_t const y = this->shared->State.y;
    uint8_t const c0 = this->shared->State.c0;
    
    if (y)
        this->bot = this->mid + 1;
    else
        this->top = this->mid;
    
    if (bpos == 0) {
        uint8_t const c1 = this->shared->State.c1;
        lstm.Perceive(c1);
        auto const& output = lstm.Predict(c1);
        memcpy(&this->probs[0], &output[0], this->alphabetSize * sizeof(float));
        this->top = this->alphabetSize - 1;
        this->bot = 0;
        
        if ((this->shared->GetOptionTrainLSTM()) && (this->shared->State.blockPos == 0)) {
            BlockType const blockType = static_cast<BlockType>(this->shared->State.blockType);
            if (blockType != pBlockType) {
                switch (blockType) {
                    case BlockType::TEXT:
                    case BlockType::TEXT_EOL:
                        modelType = LSTM::Model::Type::English;
                        break;
                    case BlockType::EXE:
                        modelType = LSTM::Model::Type::x86_64;
                        break;
                    default:
                        modelType = LSTM::Model::Type::Default;
                }
                
                if (modelType != pModelType) {
                    if ((pModelType == LSTM::Model::Type::x86_64) && (blockType == BlockType::DEFAULT)) {
                        // Skip switching
                    } else {
                        lstm.SaveModel(*repo[pModelType]);
                        lstm.LoadModel(*repo[modelType]);
                        pModelType = modelType;
                    }
                }
            }
            pBlockType = blockType;
        }
    }

    this->mid = (this->bot + this->top) >> 1;
    float prediction, num, denom;
    
    if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
        num = sum256_ps(&this->probs[this->mid + 1], this->top - this->mid, 0.f);
        denom = sum256_ps(&this->probs[this->bot], this->mid + 1 - this->bot, num);
#endif
    } else {
        num = 0.0f;
        for (size_t i = this->mid + 1; i <= this->top; i++) 
            num += this->probs[i];
        denom = num;
        for (size_t i = this->bot; i <= this->mid; i++) 
            denom += this->probs[i];
    }
    
    prediction = (denom == 0.f) ? 0.5f : num / denom;
    
    this->expected = static_cast<uint8_t>(this->bot);
    float max_prob_val = this->probs[this->bot];
    for (size_t i = this->bot + 1; i <= this->top; i++) {
        if (this->probs[i] > max_prob_val) {
            max_prob_val = this->probs[i];
            this->expected = static_cast<uint8_t>(i);
        }
    }

    this->iCtx += y;
    this->iCtx = (bpos << 8) | this->expected;
    uint32_t ctx = this->iCtx();

    int const p = std::min(std::max(std::lround(prediction * 4096.0f), 1L), 4095L);
    m.promote(stretch(p) / 2);
    m.add(stretch(p));
    m.add((p - 2048) >> 2);
    
    int const pr1 = this->apm1.p(p, (c0 << 8) | (this->shared->State.misses & 0xFF));
    int const pr2 = this->apm2.p(p, (bpos << 8) | this->expected);
    int const pr3 = this->apm3.p(pr2, ctx);
    
    m.add(stretch(pr1) >> 1);
    m.add(stretch(pr2) >> 1);
    m.add(stretch(pr3) >> 1);
    m.set((bpos << 8) | this->expected, 8 * 256);
    m.set(static_cast<uint32_t>(lstm.epoch) << 3 | bpos, 100 * 8);
}
