#pragma once

#include "../Array.hpp"
#include "../ContextMap2.hpp"
#include "../IndirectContext.hpp"
#include "../IndirectMap.hpp"
#include "../OLS_factory.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../StationaryMap.hpp"
#include "../ResidualMap.hpp"
#include "ImageModelsCommon.hpp"
#include <cstdint>

/**
 * Model for 8-bit image data (grayscale/indexed)
 */
class Image8BitModel
{
private:
  static constexpr int nSM0 = 2;
  static constexpr int nSM1 = 55;
  static constexpr int nOLS = 5;
  static constexpr int nSM = nSM0 + nSM1;
  static constexpr int nPltMaps = 4;
  static constexpr int nCM = 48 + nPltMaps;
  static constexpr int nIM = 5;

public:
  static constexpr int MIXERINPUTS =
    nSM * StationaryMap::MIXERINPUTS +
    (nOLS * 2) * ResidualMap::MIXERINPUTS +
    nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS) +
    nPltMaps * SmallStationaryContextMap::MIXERINPUTS +
    nIM * IndirectMap::MIXERINPUTS; //464
  static constexpr int MIXERCONTEXTS = 512 + 16 + 32 + 255 + 1024 + 64 + 128 + 256; /**< 2286 */
  static constexpr int MIXERCONTEXTSETS = 8;

  Shared* const shared;
  ContextMap2 cm;
  StationaryMap map[nSM];
  ResidualMap mapOLS1, mapOLS2;
  SmallStationaryContextMap pltMap[nPltMaps];  /**< palette maps, not used for grayscale images */
  IndirectMap sceneMap[nIM];
  IndirectContext<uint8_t> iCtx[nPltMaps]; /**< palette contexts, not used for grayscale images */
  Array<short> jumps{ 0x8000 };
  //pixel neighborhood
  uint8_t WWWWWW = 0, WWWWW = 0, WWWW = 0, WWW = 0, WW = 0, W = 0;
  uint8_t NWWWW = 0, NWWW = 0, NWW = 0, NW = 0, N = 0, NE = 0, NEE = 0, NEEE = 0, NEEEE = 0;
  uint8_t NNWWW = 0, NNWW = 0, NNW = 0, NN = 0, NNE = 0, NNEE = 0, NNEEE = 0;
  uint8_t NNNWW = 0, NNNW = 0, NNN = 0, NNNE = 0, NNNEE = 0;
  uint8_t NNNNW = 0, NNNN = 0, NNNNE = 0;
  uint8_t NNNNN = 0;
  uint8_t NNNNNN = 0;
  uint8_t res = 0; /**< expected residual */
  uint8_t prvFrmPx = 0; /**< corresponding pixel in previous frame */
  uint8_t prvFrmPrediction = 0; /**< prediction for corresponding pixel in previous frame */
  uint32_t lastPos = 0;
  uint32_t isGray = 0;
  int w = 0;
  int ctx = 0;
  int col = 0;
  int line = 0;
  int x = 0;
  int jump = 0;
  int framePos = 0;
  int prevFramePos = 0;
  int frameWidth = 0;
  int prevFrameWidth = 0;
  int columns[2] = { 1, 1 }, column[2]{};

  uint8_t predictions[nSM + nOLS] = { 0 };

  // Per-predictor prediction error for each decoded pixel.
  // Stores uint8 rabs(prediction - actual) for all nRM and nOLS predictors at every pixel position.
  // Read back via PredErr(ctxIndex, relX, relY) to estimate how well each predictor
  // performed on causal neighbors (W, N, NW, NE, WW, NN of the current pixel).
  // The averaged absolute error across those neighbors feeds mapR1's histogram selection,
  // giving it a spatially-local, per-predictor confidence signal:
  // low value = predictor was accurate nearby.
  // Sized in init() to cover PRED_ERR_ROWS rows: nextPowerOf2(PRED_ERR_ROWS * w * nRM).
  static constexpr size_t PRED_ERR_BUF_ROWS = 3; // three rows (including the current row) - we need to reach the prediction error of W, N, NW, NE, WW, NN
  RingBuffer<uint8_t> predErrBuf{ 0 };

  // Per-pixel accumulated decoding cost (loss), one byte per pixel.
  // Stores (loss >> 3) after each decoded byte, reflecting how hard the last pixel was
  // to compress. Read back via Ls(relX, relY); six causal neighbors (W, N, WW, NN, NW, NE)
  // are summed into lossQ, capturing local image complexity.
  // lossQ feeds mapR2's histogram selection,
  // allowing statistical maps to adapt to smooth vs. noisy/high-frequency regions.
  // Sized in init() to cover LOSS_BUF_ROWS rows: nextPowerOf2(LOSS_BUF_ROWS * w).
  static constexpr size_t LOSS_BUF_ROWS = 3; // three rows (including the current row) - we need to reach the prediction error of NN
  RingBuffer<uint8_t> lossBuf{ 0 };

  uint32_t loss = 0;  // decoding cost for the current byte, accumulated bit by bit (0..1023 over 8 bits)
  uint32_t lossQ = 0; // sum of lossBuf[] over 6 causal neighbors (W, N, WW, NN, NW, NE), capped at 639;
  // measures local image complexity: low = smooth region, high = noisy/detailed region
  uint8_t lossQ4 = 0; // lossQ quantized to 3 bits

  static constexpr float lambda[nOLS] = { 0.996f, 0.87f, 0.93f, 0.8f, 0.9f };
  static constexpr int num[nOLS] = { 32, 12, 15, 10, 14 };
  static constexpr float nu = 0.001f;
  std::unique_ptr<OLS_float> ols[nOLS];
  std::unique_ptr<OLS_float> sceneOls;

  const uint8_t* olsCtx1[32] = { &WWWWWW, &WWWWW, &WWWW, &WWW, &WW, &W, &NWWWW, &NWWW, &NWW, &NW, &N, &NE, &NEE, &NEEE, &NEEEE, &NNWWW,
                                &NNWW, &NNW, &NN, &NNE, &NNEE, &NNEEE, &NNNWW, &NNNW, &NNN, &NNNE, &NNNEE, &NNNNW, &NNNN, &NNNNE, &NNNNN,
                                &NNNNNN };
  const uint8_t* olsCtx2[12] = { &WWW, &WW, &W, &NWW, &NW, &N, &NE, &NEE, &NNW, &NN, &NNE, &NNN };
  const uint8_t* olsCtx3[15] = { &N, &NE, &NEE, &NEEE, &NEEEE, &NN, &NNE, &NNEE, &NNEEE, &NNN, &NNNE, &NNNEE, &NNNN, &NNNNE, &NNNNN };
  const uint8_t* olsCtx4[10] = { &N, &NE, &NEE, &NEEE, &NN, &NNE, &NNEE, &NNN, &NNNE, &NNNN };
  const uint8_t* olsCtx5[14] = { &WWWW, &WWW, &WW, &W, &NWWW, &NWW, &NW, &N, &NNWW, &NNW, &NN, &NNNW, &NNN, &NNNN };
  const uint8_t** olsCtxs[nOLS] = { &olsCtx1[0], &olsCtx2[0], &olsCtx3[0], &olsCtx4[0], &olsCtx5[0] };

  Image8BitModel(Shared* const sh, uint64_t size);
  ALWAYS_INLINE uint8_t Ls(int relX, int relY) const;
  ALWAYS_INLINE uint8_t GetPredErr(uint32_t ctxIndex, int relX, int relY) const;
  ALWAYS_INLINE uint32_t GetPredErrAvg(const uint32_t predictorIndex) const;
  void init(int pos);
  void setParam(int info0, uint32_t gray0);
  void mix(Mixer& m);
};
