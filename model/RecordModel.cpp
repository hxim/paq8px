#include "RecordModel.hpp"
#include "../CharacterNames.hpp"

RecordModel::RecordModel(Shared* const sh, const uint64_t size) : shared(sh),
    cm(sh,32768,3), cn(sh,32768/2,3), co(sh,32768*2,3), // cm,cn,co: memory pressure is advantageous
    cp(sh,size,16),
    maps{ /* StationaryMap :  BitsOfContext, InputBits, Scale=64, Rate=16  */
      {sh,10,8},{sh,10,8},{sh,8,8},{sh,8,8},{sh,8,8},{sh,11,1}
    },
    sMap{ /* SmallStationaryContextMap :  BitsOfContext, InputBits, Rate, Scale */
      {sh,11,1,6,86}, {sh,3,1,6,86}, {sh,19,1,5,128},
      {sh,8,8,5,64} // pos&255
    },
    iMap{ /* IndirectMap :  BitsOfContext, InputBits, Scale, Limit */
      {sh,8,8,86,255}, {sh,8,8,86,255}, {sh,8,8,86,255}
    },
    iCtx{ // IndirectContext :  BitsPerContext, InputBits
      {16,8}, {16,8}, {16,8}, {20,8}, {11,1}
    }
  {} 

void RecordModel::setParam(uint32_t fixedRecordLenght) {
  this->fixedRecordLength = fixedRecordLenght;
}

void RecordModel::mix(Mixer &m) {

  INJECT_SHARED_blockType
  bool isText = isTEXT(blockType);

  // find record length
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_buf
    INJECT_SHARED_c4
    INJECT_SHARED_pos
    int w = c4 & 0xffffU;
    int c = w & 255U;
    int d = w >> 8U;
    if(fixedRecordLength != 0) {
      rLength[0] = fixedRecordLength;
      rCount[0] = rCount[1] = rLength[1] = rLength[2] = 0;
    } else {
      //detect record length
      uint32_t r = pos - cPos1[c];
      if( r > 1 && r == cPos1[c] - cPos2[c] && r == cPos2[c] - cPos3[c] && (r > 32 || r == cPos3[c] - cPos4[c]) &&
          (r > 10 || ((c == buf(r * 5 + 1)) && c == buf(r * 6 + 1)))) {
        if( r == rLength[1] ) {
          ++rCount[0];
        } else if( r == rLength[2] ) {
          ++rCount[1];
        } else if( rCount[0] > rCount[1] ) {
          rLength[2] = r, rCount[1] = 1;
        } else {
          rLength[1] = r, rCount[0] = 1;
        }
      }

      // check candidate lengths
      for( int i = 0; i < 2; i++ ) {
        if( static_cast<int>(rCount[i]) > max(0, 12 - static_cast<int>(ilog2(rLength[i + 1])))) {
          if( rLength[0] != rLength[i + 1] ) {
            if( mayBeImg24B && rLength[i + 1] == 3 ) {
              rCount[0] >>= 1U;
              rCount[1] >>= 1U;
              continue;
            }
            if((rLength[i + 1] > rLength[0]) && (rLength[i + 1] % rLength[0] == 0)) {
              // maybe we found a multiple of the real record size..?
              // in that case, it is probably an immediate multiple (2x).
              // that is probably more likely the bigger the length, so
              // check for small lengths too
              if((rLength[0] > 32) && (rLength[i + 1] == rLength[0] * 2)) {
                rCount[0] >>= 1U;
                rCount[1] >>= 1U;
                continue;
              }
            }
            rLength[0] = rLength[i + 1];
            //printf("\nRecordModel: detected record length: %d\n",rLength[0]); // for debugging
            rCount[i] = 0;
            mayBeImg24B = (rLength[0] > 30 && (rLength[0] % 3) == 0);
            nTransition = 0;
          } else {
            // we found the same length again, that's positive reinforcement that
            // this really is the correct record size, so give it a little boost
            rCount[i] >>= 2U;
          }

          // if the other candidate record length is orders of
          // magnitude larger, it will probably never have enough time
          // to increase its counter before it's reset again. and if
          // this length is not a multiple of the other, than it might
          // really be worthwhile to investigate it, so we won't set its
          // counter to 0
          if( rLength[i + 1] << 4U > rLength[1 + (i ^ 1U)] ) {
            rCount[i ^ 1U] = 0;
          }
        }
      }
    }

    assert(rLength[0] != 0);
    col = pos % rLength[0];
    x = min(0x1F, col / max(1, rLength[0] / 32));
    N = buf(rLength[0]);
    NN = buf(rLength[0] * 2);
    NNN = buf(rLength[0] * 3);
    NNNN = buf(rLength[0] * 4);
    for( int i = 0; i < nIndContexts - 1; iCtx[i] += c, i++ ) { ;
    }
    iCtx[0] = (c << 8U) | N;
    iCtx[1] = (buf(rLength[0] - 1) << 8U) | N;
    iCtx[2] = (c << 8U) | buf(rLength[0] - 1);
    iCtx[3] = finalize64(hash(c, N, buf(rLength[0] + 1)), 20);

    /*
    Consider record structures that include fixed-length strings.
    These usually contain the text followed by either spaces or 0's,
    depending on whether they're to be trimmed or they're null-terminated.
    That means we can guess the length of the string field by looking
    for small repetitions of one of these padding bytes followed by a
    different byte. By storing the last position where this transition
    occurred, and what was the padding byte, we are able to model these
    runs of padding bytes.
    Special care is taken to skip record structures of less than 9 bytes,
    since those may be little-endian 64 bit integers. If they contain
    relatively low values (<2^40), we may consistently get runs of 3 or
    even more 0's at the end of each record, and so we could assume that
    to be the general case. But with integers, we can't be reasonably sure
    that a number won't have 3 or more 0's just before a final non-zero MSB.
    And with such simple structures, there's probably no need to be fancy
    anyway.
    */

    if( col == 0u ) {
      nTransition = 0;
    }
    if((((c4 >> 8U) == SPACE * 0x010101) && (c != SPACE)) ||
       (((c4 >> 8U) == 0u) && (c != 0) && ((padding != SPACE) || (pos - prevTransition > rLength[0])))) {
      prevTransition = pos;
      nTransition += static_cast<uint32_t>(nTransition < 31);
      padding = static_cast<uint8_t>(d);
    }

    uint64_t i = 0;

    // Set 2-3 dimensional contexts
    // assuming runLength[0]<1024; col<4096
    cm.set(hash(++i, c << 8U | (min(255, pos - cPos1[c]) >> 2U)));
    cm.set(hash(++i, w << 9U | llog(pos - wPos1[w]) >> 2));
    cm.set(hash(++i, rLength[0] | N << 10U | NN << 18U));

    cn.set(hash(++i, w | rLength[0] << 16U));
    cn.set(hash(++i, d | rLength[0] << 8U));
    cn.set(hash(++i, c | rLength[0] << 8U));

    co.set(hash(++i, c << 8U | min(255, pos - cPos1[c])));
    co.set(hash(++i, c << 17U | d << 9U | llog(pos - wPos1[w]) >> 2U));
    co.set(hash(++i, c << 8U | N));

    cp.set(hash(++i, rLength[0] | N << 10U | col << 18U));
    cp.set(hash(++i, rLength[0] | c << 10U | col << 18U));
    cp.set(hash(++i, col | rLength[0] << 12));

    if( rLength[0] > 8 ) {
      cp.set(hash(++i, min(min(0xFF, rLength[0]), pos - prevTransition), min(0x3FF, col),
                  (w & 0xF0F0U) | static_cast<uint32_t>(w == ((padding << 8U) | padding)), nTransition));
      cp.set(hash(++i, w, static_cast<uint64_t>(buf(rLength[0] + 1) == padding && N == padding), col / max(1, rLength[0] / 32)));
    } else {
      cp.set(0), cp.set(0);
    }

    cp.set(hash(++i, N | ((NN & 0xF0U) << 4U) | ((NNN & 0xE0U) << 7U) | ((NNNN & 0xE0U) << 10U) | ((col / max(1, rLength[0] / 16)) << 18U)));
    cp.set(hash(++i, (N & 0xF8U) | ((NN & 0xF8U) << 8U) | (col << 16U)));
    cp.set(hash(++i, N, NN));

    cp.set(hash(++i, col, iCtx[0]()));
    cp.set(hash(++i, col, iCtx[1]()));
    cp.set(hash(++i, col, iCtx[0]() & 0xFFU, iCtx[1]() & 0xFFU));

    cp.set(hash(++i, iCtx[2]()));
    cp.set(hash(++i, iCtx[3]()));
    cp.set(hash(++i, iCtx[1]() & 0xFFU, iCtx[3]() & 0xFFU));

    cp.set(hash(++i, N, (WxNW = c ^ buf(rLength[0] + 1))));
    cp.set(hash(++i, static_cast<int>(shared->State.Match.length2 != 0) << 8U | shared->State.Match.expectedByte, uint8_t(iCtx[1]()), N, WxNW));

    if (!isText) {
      int k = 0x300;
      if (mayBeImg24B) {
        k = (col % 3) << 8U;
        maps[0].set(clip((static_cast<uint8_t>(c4 >> 16U)) + c - (c4 >> 24U)) | k);
      }
      else {
        maps[0].set(clip(c * 2 - d) | k);
      }
      maps[1].set(clip(c + N - buf(rLength[0] + 1)) | k);
      maps[2].set(clip(N + NN - NNN));
      maps[3].set(clip(N * 2 - NN));
      maps[4].set(clip(N * 3 - NN * 3 + NNN));
      iMap[0].setDirect(N + NN - NNN);
      iMap[1].setDirect(N * 2 - NN);
      iMap[2].setDirect(N * 3 - NN * 3 + NNN);

      INJECT_SHARED_blockPos
      sMap[3].set(blockPos & 255U); // mozilla
    }

    // update last context positions
    cPos4[c] = cPos3[c];
    cPos3[c] = cPos2[c];
    cPos2[c] = cPos1[c];
    cPos1[c] = pos;
    wPos1[w] = pos;

    mxCtx = (rLength[0] > 128) ? (min(0x7F, col / max(1, rLength[0] / 128))) : col;
  }
  INJECT_SHARED_c0
  uint8_t B = c0 << (8 - bpos);
  uint32_t ctx = (N ^ B) | (bpos << 8U); // 11 bits
  INJECT_SHARED_y
  iCtx[nIndContexts - 1] += y;
  iCtx[nIndContexts - 1] = ctx;
  maps[5].set(ctx);
  maps[5].mix(m);

  sMap[0].set(ctx);
  sMap[1].set(iCtx[nIndContexts - 1]());
  sMap[2].set((ctx << 8U) | WxNW);
  sMap[0].mix(m);
  sMap[1].mix(m);
  sMap[2].mix(m);

  cm.mix(m);
  cn.mix(m);
  co.mix(m);
  cp.mix(m);
  
  if (!isText) {
    maps[0].mix(m);
    maps[1].mix(m);
    maps[2].mix(m);
    maps[3].mix(m);
    maps[4].mix(m);
    iMap[0].mix(m);
    iMap[1].mix(m);
    iMap[2].mix(m);
    sMap[3].mix(m);
  }
  
  m.set(static_cast<uint32_t>(rLength[0] > 2) * ((bpos << 7U) | mxCtx), 1024);
  m.set(((N ^ B) >> 4U) | (x << 4U), 512);
  m.set(((shared->State.Text.characterGroup) << 5U) | x, 11 * 32);
 
  shared->State.rLength = rLength[0];
}
