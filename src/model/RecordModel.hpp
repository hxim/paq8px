#pragma once

#include "../Shared.hpp"
#include "../ContextMap.hpp"
#include "../IndirectContext.hpp"
#include "../IndirectMap.hpp"
#include "../RingBuffer.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../StationaryMap.hpp"
#include "../Utils.hpp"

/**
 * Model 2-d data with fixed record length. Also order 1-2 models that include the distance to the last match.
 */
class RecordModel {
private:
  static constexpr int nCM = 3 + 3 + 3 + 16; //cm,cn,co,cp
  static constexpr int nSM = 6;
  static constexpr int nSSM = 4;
  static constexpr int nIM = 3;
  static constexpr int nIndContexts = 5;
  Shared * const shared;
  ContextMap cm, cn, co;
  ContextMap cp;
  StationaryMap maps[nSM];
  SmallStationaryContextMap sMap[nSSM];
  IndirectMap iMap[nIM];
  IndirectContext<uint16_t> iCtx[nIndContexts];
  Array<uint32_t> cPos1 {256}, cPos2 {256}, cPos3 {256}, cPos4 {256};
  Array<uint32_t> wPos1 {256 * 256}; // buf(1..2) -> last position
  uint32_t fixedRecordLength = 0; //nonzero when when record length is known 
  uint32_t rLength[3] = {2, 0, 0}; // run length and 2 candidates
  uint32_t rCount[2] = {0, 0}; // candidate counts
  uint8_t padding = 0; // detected padding byte
  uint8_t N = 0, NN = 0, NNN = 0, NNNN = 0, WxNW = 0;
  uint32_t prevTransition = 0, nTransition = 0; // position of the last padding transition
  uint32_t col = 0, mxCtx = 0, x = 0;
  bool mayBeImg24B = false;

public:
  static constexpr int MIXERINPUTS =
    nCM * ContextMap::MIXERINPUTS +
    nSM * StationaryMap::MIXERINPUTS + 
    nSSM * SmallStationaryContextMap::MIXERINPUTS +
    nIM * IndirectMap::MIXERINPUTS; // 157
  static constexpr int MIXERCONTEXTS = 1024 + 512 + 11 * 32; //1888
  static constexpr int MIXERCONTEXTSETS = 3;
  RecordModel(Shared* const sh, uint64_t size);
  void setParam(uint32_t fixedRecordLenght);
  void mix(Mixer &m);
};
