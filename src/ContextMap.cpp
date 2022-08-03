#include "ContextMap.hpp"

ContextMap::ContextMap(const Shared* const sh, uint64_t m, const int contexts) : 
  shared(sh), C(contexts), t(m >> 6), cp(contexts), cxt(contexts), chk(contexts), runP(contexts),
  sm(sh, contexts, 256, 1023, StateMapType::BitHistory), cn(0),
  mask(uint32_t(t.size() - 1)), hashBits(ilog2(mask + 1)), validFlags(0) {
  assert(m >= 64 && isPowerOf2(m));
  assert(C <= (int) sizeof(validFlags) * 8); // validFlags is 64 bits - it can't support more than 64 contexts
}

void ContextMap::set(const uint64_t cx) {
  assert(cn >= 0 && cn < C);
  const uint32_t ctx = cxt[cn] = finalize64(cx, hashBits);
  const uint16_t checksum = chk[cn] = checksum16(cx, hashBits);
  uint8_t* base = cp[cn] = &t[ctx].find(checksum, &rnd)->bitState;
  runP[cn] = base + 3;
  // update pending bit histories for bits 2-7
  if( base[3] == 2 ) {
    const int c = base[4] + 256;
    uint8_t *p = &t[(ctx + (c >> 6)) & mask].find(checksum, &rnd)->bitState;
    p[0] = 1 + ((c >> 5) & 1);
    p[1 + ((c >> 5) & 1)] = 1 + ((c >> 4) & 1);
    p[3 + ((c >> 4) & 3)] = 1 + ((c >> 3) & 1);
    p = &t[(ctx + (c >> 3)) & mask].find(checksum, &rnd)->bitState;
    p[0] = 1 + ((c >> 2) & 1);
    p[1 + ((c >> 2) & 1)] = 1 + ((c >> 1) & 1);
    p[3 + ((c >> 1) & 3)] = 1 + (c & 1);
  }
  cn++;
  validFlags = (validFlags << 1) + 1;
}

void ContextMap::skip() {
  assert(cn >= 0 && cn < C);
  cn++;
  validFlags <<= 1;
}

void ContextMap::update() {
  INJECT_SHARED_y
  INJECT_SHARED_bpos
  INJECT_SHARED_c1
  INJECT_SHARED_c0
  for( int i = 0; i < cn; ++i ) {
    if(((validFlags >> (cn - 1 - i)) & 1) != 0 ) {

      // update bit history state byte
      if( cp[i] != nullptr ) {
        const uint32_t pis = (0b0100110100010011 >> (bpos << 1)) & 0x03;
        StateTable::update(cp[i] + pis + (((bpos == 0 ? c1 : c0) >> 1) & pis), y, rnd);
      }

      // update context pointers
      if ((bpos == 2 && runP[i][0] == 0) || (cp[i] == nullptr)) {
        cp[i] = nullptr; //defer updating in 2nd and 3rd slots (i.e. bits 2,3,4 and 5,6,7) when context is seen for the first time
      } 
      else if( bpos == 2 || bpos == 5 ) {
        const uint16_t checksum = chk[i];
        const uint32_t ctx = cxt[i];
        cp[i] = &t[(ctx + c0) & mask].find(checksum, &rnd)->bitState;
      }
      if (bpos==0) {
        // update run count of previous context
        if( runP[i][0] == 0 ) { // new context
          runP[i][0] = 2, runP[i][1] = c1;
        } else if( runP[i][1] != c1 ) { // different byte in context
          runP[i][0] = 1, runP[i][1] = c1;
        } else if( runP[i][0] < 254 ) { // same byte in context
          runP[i][0] += 2;
        } else if( runP[i][0] == 255 ) {
          runP[i][0] = 128;
        }
      }
    }
  }
  if( bpos == 0 ) {
    cn = 0;
    validFlags = 0;
  }
}

void ContextMap::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  sm.subscribe();
  INJECT_SHARED_bpos
  INJECT_SHARED_c0
  for( int i = 0; i < cn; ++i ) {
    if(((validFlags >> (cn - 1 - i)) & 1) != 0 ) {
      // predict from last byte in context
      if((runP[i][1] + 256) >> (8 - bpos) == c0 ) {
        int rc = runP[i][0]; // count*2, +1 if 2 different bytes seen
        int sign = (runP[i][1] >> (7 - bpos) & 1) * 2 - 1; // predicted bit + for 1, - for 0
        int c = ilog->log(rc + 1) << (2 + (~rc & 1));
        m.add(sign * c);
      } else {
        m.add(0); //p=0.5
      }

      // predict from bit context
      const uint32_t pis = (0b1101001101000100 >> (bpos << 1)) & 0x03;
      const int s = cp[i] != nullptr ? *(cp[i]+ pis+(c0&pis)) : 0;
      if( s == 0 ) { //skip context
        sm.skip(i);
        m.add(0);
        m.add(0);
        m.add(0);
        m.add(0);
      } else {
        const int p1 = sm.p2(i, s);
        const int st = stretch(p1) >> 2;
        const int contextIsYoung = int(s <= 2);
        m.add(st >> contextIsYoung);
        m.add((p1 - 2048) >> 3);
        const int n0 = -!StateTable::getNextState(s, 2);
        const int n1 = -!StateTable::getNextState(s, 3);
        m.add((n0 | n1) & st); // when both counts are nonzero add(0) otherwise add(st)
        const int p0 = 4095 - p1;
        m.add(((p1 & n0) - (p0 & n1)) >> 4);
      }
    } else { //skipped context
      sm.skip(i);
      m.add(0);
      m.add(0);
      m.add(0);
      m.add(0);
      m.add(0);
    }
  }
}
