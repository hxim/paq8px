#pragma once

#include "../Shared.hpp"
#include "../Mixer.hpp"
#include "../APM.hpp"
#include "../IndirectContext.hpp"
#include "../UpdateBroadcaster.hpp"
#include "Lstm.hpp"
#include "ByteModelToBitModel.hpp"
#include <cstdint>

/**
 * Encapsulating the LSTM model into a paq-compatible container
 */
class LstmModelContainer : IPredictor {
private:
  Shared* const shared;

  SIMDType simd;
  LstmShape shape;
  Lstm lstm;

  float* probs;
  ByteModelToBitModel byteModelToBitModel;
  APM apm1, apm2, apm3;
  IndirectContext<std::uint16_t> iCtx;
  uint8_t expectedByte;

public:
  static constexpr int MIXERINPUTS = 5;
  static constexpr int MIXERCONTEXTS = 8 * 256 + 8 * 100;
  static constexpr int MIXERCONTEXTSETS = 2;
  static constexpr size_t alphabetSize = 1llu << 8;

  explicit LstmModelContainer(Shared* const sh);

  void next();
  float getp();
  void mix(Mixer& m);
  void update() override;

  void SaveModelParameters(FILE* file);
  void LoadModelParameters(FILE* file);

  static int GetNumberOfTrainableParameters(Shared* sh) {
    int vocabulary_size = alphabetSize;
    int hidden_size = sh->LstmSettings.hidden_size;
    int num_layers = sh->LstmSettings.num_layers;

    int total = 0;

    // Parameters for each LSTM layer
    for (int layer = 0; layer < num_layers; layer++) {
      int component_input_dim = (layer > 0) ? (2 * hidden_size) : hidden_size;

      // Each layer has 3 components (forget gate, cell candidate, output gate)
      int params_per_component =
        (hidden_size * vocabulary_size) +  // symbol_embeddings
        (hidden_size * component_input_dim) +  // weights
        hidden_size +  // bias
        hidden_size +  // gamma (RMSNorm)
        hidden_size;   // beta (RMSNorm)

      total += 3 * params_per_component;
    }

    // Output layer parameters
    total += vocabulary_size * (hidden_size * num_layers);  // output_weights
    total += vocabulary_size;  // output_bias

    return total;
  }
};
