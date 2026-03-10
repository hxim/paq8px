#pragma once

#include "../ContextMap2.hpp"
#include "../OLS_factory.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../LargeStationaryMap.hpp"
#include "../StationaryMap.hpp"
#include "../ResidualMap.hpp"
#include "ImageModelsCommon.hpp"
#include <cmath>

/**
 * Model 24/32-bit image data
 */
class Image24BitModel {
private:
  static constexpr int nDM = 102;
  static constexpr int nOLS = 6;
  static constexpr int nSM = nOLS;
  static constexpr int nLSM = 43;
  static constexpr int nCM = 30;
  Ilog *ilog = &Ilog::getInstance();

public:
  static constexpr int MIXERINPUTS =
    nLSM * LargeStationaryMap::MIXERINPUTS +
    (nDM * 3) * ResidualMap::MIXERINPUTS +
    nSM * StationaryMap::MIXERINPUTS +
    nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS); // 909
  static constexpr int MIXERCONTEXTS = (1 + 4 * 8) + (16 * 4) + 256 + 256 + 512 + 2048 + (8 * 32) + (6 * 64) + 1024 + 8192 + 256; // 13281
  static constexpr int MIXERCONTEXTSETS = 11;

  Shared * const shared;

  // probability maps
  ContextMap2 cm;
  ResidualMap mapR1, mapR2, mapR3;
  LargeStationaryMap mapL;
  StationaryMap map[nSM];

  // Per-predictor signed prediction error for each decoded pixel.
  // Stores int8 (prediction - actual) for all nDM predictors at every pixel position.
  // Read back via PredErr(ctxIndex, relX, relY) to estimate how well each predictor
  // performed on causal neighbors (W, N, NW, NE of the current pixel, same color plane).
  // The averaged absolute error across those neighbors feeds mapR1's histogram selection,
  // giving it a spatially-local, per-predictor confidence signal:
  // low value = predictor was accurate nearby.
  // Sized in init() to cover PRED_ERR_ROWS rows: nextPowerOf2(PRED_ERR_ROWS * w * nDM).
  static constexpr size_t PRED_ERR_BUF_ROWS = 2; // two rows (including the current row) - we need to reach the predistion error of N, NW, NE
  RingBuffer<uint8_t> predErrBuf{ 0 };

  // Per-pixel accumulated decoding cost (loss), one byte per pixel per color plane.
  // Stores (loss >> 3) after each decoded byte, reflecting how hard the last pixel was
  // to compress. Read back via Ls(relX, relY); six causal neighbors (W, N, WW, NN, NW, NE)
  // are summed into lossQ, capturing local image complexity.
  // lossQ feeds mapR2's histogram selection,
  // allowing statistical maps to adapt to smooth vs. noisy/high-frequency regions.
  // Sized in init() to cover LOSS_BUF_ROWS rows: nextPowerOf2(LOSS_BUF_ROWS * w).
  static constexpr size_t LOSS_BUF_ROWS = 3; // three rows (including the current row) - we need to reach the predistion error of NN
  RingBuffer<uint8_t> lossBuf{ 0 };

  uint32_t loss = 0;  // decoding cost for the current byte, accumulated bit by bit (0..255 over 8 bits)
  uint32_t lossQ = 0; // sum of lossBuf[] over 6 causal neighbors (W, N, WW, NN, NW, NE), capped at 639;
                      // measures local image complexity: low = smooth region, high = noisy/detailed region

  // pixel neighborhood
  uint8_t WWWWWW = 0, WWWWW = 0, WWWW = 0, WWW = 0, WW = 0, W = 0;
  uint8_t NWWWW = 0, NWWW = 0, NWW = 0, NW = 0, N = 0, NE = 0, NEE = 0, NEEE = 0, NEEEE = 0;
  uint8_t NNNWWW = 0, NNWWW = 0, NNWW = 0, NNW = 0, NN = 0, NNE = 0, NNEE = 0, NNEEE = 0;
  uint8_t NNNWW = 0, NNNW = 0, NNN = 0, NNNE = 0, NNNEE = 0, NNNEEE = 0;
  uint8_t NNNNW = 0, NNNN = 0, NNNNE = 0;
  uint8_t NNNNN = 0;
  uint8_t NNNNNN = 0;
  uint8_t WWp1 = 0, Wp1 = 0, p1 = 0, NWp1 = 0, Np1 = 0, NEp1 = 0, NNp1 = 0;
  uint8_t WWp2 = 0, Wp2 = 0, p2 = 0, NWp2 = 0, Np2 = 0, NEp2 = 0, NNp2 = 0;

  int info = 0;
  uint32_t alpha = 0;
  int color = 0;
  int stride = 3;
  uint32_t ctx[2]{}, padding = 0, x = 0, w = 0, line = 0;
  uint32_t lastPos = 0;
  int columns[2] = {1, 1}, column[2]{};
  uint32_t predictions[nDM] = { 0 };
  uint8_t pOLS[nOLS] = {0}; // Clipped OLS predictions (one per OLS predictor), used as input to StationaryMap.

  static constexpr float lambda[nOLS] = {0.98f, 0.87f, 0.9f, 0.8f, 0.9f, 0.7f};
  static constexpr int num[nOLS] = {32, 12, 15, 10, 14, 8};
  static constexpr float nu = 0.001f;
  std::unique_ptr<OLS_float> ols[nOLS][4]; // 4: for RGBA color components

  const uint8_t *olsCtx1[32] = {&WWWWWW, &WWWWW, &WWWW, &WWW, &WW, &W, &NWWWW, &NWWW, &NWW, &NW, &N, &NE, &NEE, &NEEE, &NEEEE, &NNWWW,
                                &NNWW, &NNW, &NN, &NNE, &NNEE, &NNEEE, &NNNWW, &NNNW, &NNN, &NNNE, &NNNEE, &NNNNW, &NNNN, &NNNNE, &NNNNN,
                                &NNNNNN};
  const uint8_t *olsCtx2[12] = {&WWW, &WW, &W, &NWW, &NW, &N, &NE, &NEE, &NNW, &NN, &NNE, &NNN};
  const uint8_t *olsCtx3[15] = {&N, &NE, &NEE, &NEEE, &NEEEE, &NN, &NNE, &NNEE, &NNEEE, &NNN, &NNNE, &NNNEE, &NNNN, &NNNNE, &NNNNN};
  const uint8_t *olsCtx4[10] = {&N, &NE, &NEE, &NEEE, &NN, &NNE, &NNEE, &NNN, &NNNE, &NNNN};
  const uint8_t *olsCtx5[14] = {&WWWW, &WWW, &WW, &W, &NWWW, &NWW, &NW, &N, &NNWW, &NNW, &NN, &NNNW, &NNN, &NNNN};
  const uint8_t *olsCtx6[8] = {&WWW, &WW, &W, &NNN, &NN, &N, &p1, &p2};
  const uint8_t **olsCtxs[nOLS] = {&olsCtx1[0], &olsCtx2[0], &olsCtx3[0], &olsCtx4[0], &olsCtx5[0], &olsCtx6[0]};

  Image24BitModel(Shared* const sh, uint64_t size);
  void update();

  ALWAYS_INLINE uint8_t Px(int relX, int relY, int colorShift) const;
  ALWAYS_INLINE uint8_t Ls(int relX, int relY) const;
  ALWAYS_INLINE uint8_t GetPredErr(uint32_t ctxIndex, int relX, int relY) const;
  ALWAYS_INLINE void MakePrediction(int i, uint8_t base1, uint8_t base2, int prediction);
  ALWAYS_INLINE void MakePredictionC(int i, int prediction);
  ALWAYS_INLINE void MakePredictionTrend(int i, int base1, int other1, int base2);
  ALWAYS_INLINE void MakePredictionSmooth(int i, int base1, int other1, int base2);

  /**
   * New image.
   */
  void init();
  void setParam(int width, uint32_t alpha0);
  void mix(Mixer &m);
};
