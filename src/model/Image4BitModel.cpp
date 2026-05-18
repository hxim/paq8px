#include "Image4BitModel.hpp"
#include "../Stretch.hpp"


Image4BitModel::Image4BitModel(const Shared* const sh, const uint64_t size) : shared(sh),
  rnd(),
  hashTable(size >> 6),
  mask(uint32_t(hashTable.size() - 1)),
  hashBits(ilog2(mask + 1)),
  hashes(S),
  sm {sh, S, 256, 1023, StateMapType::BitHistory},
  mapL (sh, S, 22)
{}

void Image4BitModel::setParam(int widthInBytes) {
  w = widthInBytes;
}

void Image4BitModel::update() {
  INJECT_SHARED_y
  for( int i = 0; i < S; i++ ) {
    StateTable::update(cp[i], y, rnd);
  }
  INJECT_SHARED_bpos
  if (bpos == 0 || bpos == 4) {
    col++;
    if (col == w * 2) {
      col = 0;
      line++;
    }
  }
}

void Image4BitModel::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);

  INJECT_SHARED_bpos
  INJECT_SHARED_c0
  if( bpos == 0 || bpos == 4 ) {
    WW = W;
    NWW = NW;
    NW = N;
    N = NE;
    NE = NEE;
    NNWW = NWW;
    NNW = NN;
    NN = NNE;
    NNE = NNEE;
    INJECT_SHARED_buf
    if( bpos == 0 ) {
      INJECT_SHARED_c1
      W = c1 & 0xF;
      NEE = buf(w - 1) >> 4;
      NNEE = buf(w * 2 - 1) >> 4;
    } else {
      W = c0 & 0xF;
      NEE = buf(w - 1) & 0xF;
      NNEE = buf(w * 2 - 1) & 0xF;
    }
    run = (W != WW || col == 0) ? (prevColor = WW, 0) : min(0xFFF, run + 1);
  }

  const uint32_t msb = 1 << (bpos & 3); //bpos = {0,1,2,3 / 4,5,6,7} -> 1,2,4,8
  const uint32_t px = msb | (c0 & (msb - 1)); // 0001, 001x, 01xx, 1xxx

  uint64_t i = 0;
  uint64_t j = static_cast<uint64_t>(px * 256);

  hashes[i++] = hash(++j, W, NW, N);
  hashes[i++] = hash(++j, N, min(0xFFF, col / 8));
  hashes[i++] = hash(++j, W, NW, N, NN, NE);
  hashes[i++] = hash(++j, W, N, NE + NNE * 16, NEE + NNEE * 16);
  hashes[i++] = hash(++j, W, N, NW + NNW * 16, NWW + NNWW * 16);
  hashes[i++] = hash(++j, W, ilog2(run + 1), prevColor, col / max(1, w / 2));
  hashes[i++] = hash(++j, NE, min(0x3FF, (col + line) / max(1, w * 8)));
  hashes[i++] = hash(++j, NW, (col - line) / max(1, w * 8));
  hashes[i++] = hash(++j, WW * 16 + W, NN * 16 + N, NNWW * 16 + NW);
  hashes[i++] = hash(++j, N, NN);
  hashes[i++] = hash(++j, W, WW);
  hashes[i++] = hash(++j, W, NE);
  hashes[i++] = hash(++j, WW, NN, NEE);
  hashes[i++] = hash(++j);

  assert(i == S);
  assert(j == px * 256 + S);
     
  for (size_t i = 0; i < S; i++) {
    uint64_t contexthash = hashes[i];
    mapL.set(contexthash);
    const uint32_t ctx = finalize64(contexthash, hashBits);
    const uint16_t chk = checksum16(contexthash, hashBits);
    cp[i] = &hashTable[ctx].find(chk, &rnd)->bitState;
  }

  mapL.mix(m);

  sm.subscribe();
  for( int i = 0; i < S; i++ ) {
    const uint8_t state = *cp[i];
    const int n0 = StateTable::getNextState(state, 2);
    const int n1 = StateTable::getNextState(state, 3);
    const int p1 = sm.p2(i, state);
    const int st = stretch(p1) >> 1;
    m.add(st);
    m.add((p1 - 2048) >> 2);
    const int bitIsUncertain = int(n0 != 0 && n1 != 0);
    m.add((bitIsUncertain - 1) & st); // when both counts are nonzero add(0) otherwise add(st)
  }

  m.set((W << 4) | px, 256);
  m.set(min(31, col / max(1, w / 16)) | (N << 5), 512);
  m.set((bpos & 3) | (W << 2) | (min(7, ilog2(run + 1)) << 6), 512);
  m.set(W | (NE << 4) | ((bpos & 3) << 8), 1024);
  m.set(px, 16);
  m.set(0, 1);
}
