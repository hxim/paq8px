#include "ContextMap2.hpp"

ContextMap2::ContextMap2(const Shared* const sh, const uint64_t size, const uint32_t contexts, const int scale) : shared(sh), 
        C(contexts), contextInfoList(contexts), hashTable(size >> 6),
        runMap(sh, contexts, (1 << 12), 127, StateMap::Run),         /* StateMap : s, n, lim, init */ // 63-255
        stateMap(sh, contexts, (1 << 8), 511, StateMap::BitHistory), /* StateMap : s, n, lim, init */ // 511-1023
        bhMap8B(sh, contexts, (1 << 8), 511, StateMap::Generic),     /* StateMap : s, n, lim, init */ // 511-1023
        bhMap12B(sh, contexts, (1 << 12), 511, StateMap::Generic),   /* StateMap : s, n, lim, init */ // 255-1023
        index(0), mask(uint32_t(hashTable.size() - 1)), hashBits(ilog2(mask + 1)), scale(scale), contextflagsAll(0) {
#ifdef VERBOSE
  printf("Created ContextMap2 with size = %" PRIu64 ", contexts = %d, scale = %d\n", size, contexts, scale);
#endif
  assert(size >= 64 && isPowerOf2(size));
  static_assert(sizeof(Bucket16 <HashElementForContextMap, 7>) == (sizeof(HashElementForContextMap) + 2) * 7, "Something is wrong, please pack that struct");
}

void ContextMap2::updatePendingContextsInSlot(HashElementForContextMap* const p, uint32_t c) {
  // in case of a collision updating (mixing) is slightly better (but slightly slower) then resetting, so we update
  StateTable::update(&p->bitState, (c >> 2) & 1, rnd);
  StateTable::update(&p->bitState0 + ((c >> 2) & 1), (c >> 1) & 1, rnd);
  StateTable::update(&p->bitStates.bitState00 + ((c >> 1) & 3), c & 1, rnd);
}

void ContextMap2::updatePendingContexts(uint32_t ctx, uint16_t checksum, uint32_t c) {
  // update pending bit histories for bits 2, 3, 4
  HashElementForContextMap* const p1A = hashTable[(ctx + (c >> 6)) & mask].find(checksum, &rnd);
  updatePendingContextsInSlot(p1A, c >> 3);
  // update pending bit histories for bits 5, 6, 7
  HashElementForContextMap* const p1B = hashTable[(ctx + (c >> 3)) & mask].find(checksum, &rnd);
  updatePendingContextsInSlot(p1B, c);
}

void ContextMap2::set(uint8_t ctxflags, const uint64_t contexthash) {
  assert(index >= 0 && index < C);
  ContextInfo *contextInfo = &contextInfoList[index];
  const uint32_t ctx = contextInfo->tableIndex = finalize64(contexthash, hashBits);
  const uint16_t chk = contextInfo->tableChecksum = checksum16(contexthash, hashBits);
  HashElementForContextMap* const slot0 =  hashTable[ctx].find(chk, &rnd);
  contextInfo->slot0 = slot0;
  contextInfo->slot012 = slot0;

  if (slot0->bitState <= 6) { // while constructing statistics for the first 3 bytes (states: 0; 1-2; 3-6) defer updating bit statistics in slot1 and slot2
    ctxflags |= CM_DEFERRED;
  }
  else if (slot0->bitState <= 14) { // the first 3 bytes in this context are now known, it's time to update pending bit histories
    if (slot0->byteStats.runcount == 2) {
      updatePendingContexts(ctx, chk, slot0->byteStats.byte2 + 256);
      updatePendingContexts(ctx, chk, slot0->byteStats.byte1 + 256);
      updatePendingContexts(ctx, chk, slot0->byteStats.byte1 + 256);
    }
    else {
      updatePendingContexts(ctx, chk, slot0->byteStats.byte3 + 256);
      updatePendingContexts(ctx, chk, slot0->byteStats.byte2 + 256);
      updatePendingContexts(ctx, chk, slot0->byteStats.byte1 + 256);
    }
  }
  
  contextInfo->flags = ctxflags;
  contextflagsAll |= ctxflags;
  index++;
}


void ContextMap2::skip(const uint8_t ctxflags) {
  assert(index >= 0 && index < C);
  contextInfoList[index].flags = ctxflags | CM_SKIPPED_CONTEXT;
  index++;
}

void ContextMap2::skipn(const uint8_t ctxflags, const int n) {
  for (int i = 0; i < n; i++)
    skip(ctxflags);
}

ALWAYS_INLINE
size_t ContextMap2::getStateByteLocation(const uint32_t bpos, const uint32_t c0) {
  uint32_t pis = 0; //state byte position in slot
  if (false) {
    // this version is for readability
    switch (bpos) {
    case 0: //slot0
      pis = 0;
      break;
    case 1: //slot0
      pis = 1 + (c0 & 1);
      break;
    case 2: //slot1
      pis = 0;
      break;
    case 3: //slot1
      pis = 1 + (c0 & 1);
      break;
    case 4: //slot1
      pis = 3 + (c0 & 3);
      break;
    case 5: //slot2
      break;
    case 6: //slot2
      pis = 1 + (c0 & 1);
      break;
    case 7: //slot2
      pis = 3 + (c0 & 3);
      break;
    }
  }
  else {
    // this is a speed optimized (branchless) version of the above
    const uint32_t smask = (UINT32_C(0x31031010) >> (bpos << 2)) & 0x0F;
    pis = smask + (c0 & smask);
  }

  return pis;
}


void ContextMap2::update() {
  INJECT_SHARED_y
  INJECT_SHARED_bpos
  INJECT_SHARED_c1
  INJECT_SHARED_c0
  for( uint32_t i = 0; i < index; i++ ) {
    ContextInfo* contextInfo = &contextInfoList[i];
    if((contextInfo->flags & CM_SKIPPED_CONTEXT) == 0) {
      uint8_t* pState = &contextInfo->slot012->bitState + getStateByteLocation((bpos - 1) & 7, (bpos == 0 ? c1 + 256u : c0) >> 1);
      StateTable::update(pState, y, rnd);
      
      assume(bpos >= 0 && bpos <= 7);
      if (bpos == 0) {
        // update byte history and run statistics
        if (contextInfo->slot0->bitState < 3) {
          contextInfo->slot0->byteStats.byte3 = contextInfo->slot0->byteStats.byte2 = contextInfo->slot0->byteStats.byte1 = c1;
          contextInfo->slot0->byteStats.runcount = 1;
        }
        else {
          const bool isMatch = contextInfo->slot0->byteStats.byte1 == c1;
          if (isMatch) {
            uint8_t runCount = contextInfo->slot0->byteStats.runcount;
            if (runCount < 255) {
              contextInfo->slot0->byteStats.runcount = runCount + 1;
            }
          }
          else {
            // shift byte candidates
            contextInfo->slot0->byteStats.runcount = 1;
            contextInfo->slot0->byteStats.byte3 = contextInfo->slot0->byteStats.byte2;
            contextInfo->slot0->byteStats.byte2 = contextInfo->slot0->byteStats.byte1;
            contextInfo->slot0->byteStats.byte1 = c1; //last byte seen
          }
        }
      }
      else if( bpos==2 || bpos==5 ) {
        if (contextInfo->flags & CM_DEFERRED) { //when in deferred mode...
          // ...reconstruct bit states in temporary location from last seen bytes 

          memset(&contextInfo->bitStateTmp, 0, 7);
          contextInfo->slot012 = &contextInfo->bitStateTmp;
          
          const uint8_t bit0state = contextInfo->slot0->bitState;
          if (bit0state >= 3) { // at least 1 byte was seen
            const uint8_t byte2 = bit0state >= 7 ? contextInfo->slot0->byteStats.byte2 : contextInfo->slot0->byteStats.byte1;
            const uint8_t mask = ((1 << bpos) - 1);
            const int shift = 8 - bpos;
            if (((c0 ^ (byte2 >> shift)) & mask) == 0) { // last 2/5 bits must match otherwise it's not the current slot location
              updatePendingContextsInSlot(&contextInfo->bitStateTmp, byte2 >> (shift - 3)); // simulate the current states at the temporary location
            }
            if (bit0state >= 7) { // at least 2 bytes were seen
              const uint8_t byte1 = contextInfo->slot0->byteStats.byte1;
              if (((c0 ^ (byte1 >> shift)) & mask) == 0) { // last 2/5 bits must match otherwise it's not the current slot location
                updatePendingContextsInSlot(&contextInfo->bitStateTmp, byte1 >> (shift - 3)); // simulate the current states at the temporary location
              }
            }
          }
        }
        else {
          const uint32_t ctx = contextInfo->tableIndex;
          const uint16_t chk = contextInfo->tableChecksum;
          contextInfo->slot012 = hashTable[(ctx + c0) & mask].find(chk, &rnd);
        }
      }
    }
  }
  if( bpos == 0 ) {
    index = 0;
    contextflagsAll = 0;
  } // start over
}

void ContextMap2::setScale(const int Scale) { scale = Scale; }

void ContextMap2::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  stateMap.subscribe();
  if ((contextflagsAll & CM_USE_RUN_STATS) != 0)
    runMap.subscribe();
  if ((contextflagsAll & CM_USE_BYTE_HISTORY) != 0) {
    bhMap8B.subscribe();
    bhMap12B.subscribe();
  }

  order = 0;
  INJECT_SHARED_bpos
  INJECT_SHARED_c0
  for( uint32_t i = 0; i < index; i++ ) {
    ContextInfo* contextInfo = &contextInfoList[i];
    if((contextInfo->flags & CM_SKIPPED_CONTEXT) == 0 ) {

      uint8_t* pState = &contextInfo->slot012->bitState + getStateByteLocation(bpos, c0);
      const int state = *pState;
      const int n0 = StateTable::next(state, 2);
      const int n1 = StateTable::next(state, 3);
      const int bitIsUncertain = int(n0 != 0 && n1 != 0);

      // predict from last byte(s) in context
      uint8_t byteState = contextInfo->slot0->bitState;
      const bool complete1 = (byteState >= 3) || (byteState >= 1 && bpos == 0);
      const bool complete2 = (byteState >= 7) || (byteState >= 3 && bpos == 0);
      const bool complete3 = (byteState >= 15) || (byteState >= 7 && bpos == 0);
      if((contextInfo->flags & CM_USE_RUN_STATS) != 0 ) {
        const int bp = (UINT32_C(0x33322210) >> (bpos << 2)) & 0xF; // {bpos:0}->0  {bpos:1}->1  {bpos:2,3,4}->2  {bpos:5,6,7}->3
        bool skipRunMap = true;
        if( complete1 ) {
          if(((contextInfo->slot0->byteStats.byte1 + 256u) >> (8 - bpos)) == c0 ) { // 1st candidate (last byte seen) matches
            const int predictedBit = (contextInfo->slot0->byteStats.byte1 >> (7 - bpos)) & 1;
            const int byte1IsUncertain = static_cast<const int>(contextInfo->slot0->byteStats.byte2 != contextInfo->slot0->byteStats.byte1);
            const int runCount = contextInfo->slot0->byteStats.runcount; // 1..255
            m.add(stretch(runMap.p2(i, runCount << 4 | bp << 2 | byte1IsUncertain << 1 | predictedBit)) >> (1 + byte1IsUncertain));
            skipRunMap = false;
          } else if( complete2 && ((contextInfo->slot0->byteStats.byte2 + 256u) >> (8 - bpos)) == c0 ) { // 2nd candidate matches
            const int predictedBit = (contextInfo->slot0->byteStats.byte2 >> (7 - bpos)) & 1;
            const int byte2IsUncertain = static_cast<const int>(contextInfo->slot0->byteStats.byte3 != contextInfo->slot0->byteStats.byte2);
            m.add(stretch(runMap.p2(i, bitIsUncertain << 1 | predictedBit)) >> (2 + byte2IsUncertain));
            skipRunMap = false;
          }
          // remark: considering the 3rd byte is not beneficial in most cases, except for some 8bpp images
        }
        if( skipRunMap ) {
          runMap.skip(i);
          m.add(0);
        }
      }
      // predict from bit context
      if( state == 0 ) {
        stateMap.skip(i);
        m.add(0);
        m.add(0);
        m.add(0);
        m.add(0);
      } else {
        const int p1 = stateMap.p2(i, state);
        const int st = (stretch(p1) * scale) >> 8;
        const int contextIsYoung = int(state <= 2);
        m.add(st >> contextIsYoung);
        m.add(((p1 - 2048) * scale) >> 9);
        m.add((bitIsUncertain - 1) & st); // when both counts are nonzero add(0) otherwise add(st)
        const int p0 = 4095 - p1;
        m.add((((p1 & (-!n0)) - (p0 & (-!n1))) * scale) >> 10);
        order++;
      }

      if((contextInfo->flags & CM_USE_BYTE_HISTORY) != 0 ) {
        const int bhBits = 
          (((contextInfo->slot0->byteStats.byte1 >> (7 - bpos)) & 1)) |
          (((contextInfo->slot0->byteStats.byte2 >> (7 - bpos)) & 1) << 1) |
          (((contextInfo->slot0->byteStats.byte3 >> (7 - bpos)) & 1) << 2);

        int bhState = 0; // 4 bit
        if( complete3 ) {
          bhState = 8 | (bhBits); //we have seen 3 bytes (at least)
        } else if( complete2 ) {
          bhState = 4 | (bhBits & 3); //we have seen 2 bytes
        } else if( complete1 ) {
          bhState = 2 | (bhBits & 1); //we have seen 1 byte only
        }
        //else new context (bhState=0)

        const uint8_t stateGroup = StateTable::group(state); //0..31
        m.add(stretch(bhMap8B.p2(i, bitIsUncertain << 7 | (bhState << 3) | bpos)) >> 2); // using bitIsUncertain is generally beneficial except for some 8bpp image (noticeable loss)
        m.add(stretch(bhMap12B.p2(i, stateGroup << 7 | (bhState << 3) | bpos)) >> 2);
      }
    } else { //skipped context
      if((contextInfo->flags & CM_USE_RUN_STATS) != 0 ) {
        runMap.skip(i);
        m.add(0);
      }
      if((contextInfo->flags & CM_USE_BYTE_HISTORY) != 0 ) {
        bhMap8B.skip(i);
        m.add(0);
        bhMap12B.skip(i);
        m.add(0);
      }
      stateMap.skip(i);
      m.add(0);
      m.add(0);
      m.add(0);
      m.add(0);
    }
  }
}
