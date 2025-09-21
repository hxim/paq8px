#include "MatchInfo.hpp"

#include <cassert>
#include <algorithm>

bool MatchInfo::isInNoMatchMode() const {
  return length == 0 && !delta && lengthBak == 0;
}

bool MatchInfo::isInPreRecoveryMode() const {
  return length == 0 && !delta && lengthBak != 0;
}

bool MatchInfo::isInRecoveryMode() const {
  return length != 0 && lengthBak != 0;
}

uint32_t MatchInfo::recoveryModePos() const {
  assert(isInRecoveryMode()); //must be in recovery mode
  return length - lengthBak;
}

uint64_t MatchInfo::prio() const {
  return
    static_cast<uint64_t>(length != 0) << 49 | //normal mode (match)
    static_cast<uint64_t>(delta) << 48 | //delta mode
    static_cast<uint64_t>(delta ? lengthBak : length) << 32 | //the longer wins
    static_cast<uint64_t>(index); //the more recent wins
}

bool MatchInfo::isBetterThan(const MatchInfo* other) const {
  return this->prio() > other->prio();
}

void MatchInfo::update(Shared* shared, uint32_t minlen_rm) {
  if constexpr (false) {
    INJECT_SHARED_bpos
      INJECT_SHARED_blockPos
      printf("- pos %d %d  index %d  length %d  lengthBak %d  delta %d\n", blockPos, bpos, index, length, lengthBak, delta ? 1 : 0);
  }
  INJECT_SHARED_buf
    INJECT_SHARED_bpos
    if (length != 0) {
      const int expectedBit = (expectedByte >> ((8 - bpos) & 7)) & 1;
      INJECT_SHARED_y
        if (y != expectedBit) {
          if (isInRecoveryMode()) { // another mismatch in recovery mode -> give up
            lengthBak = 0;
            indexBak = 0;
          }
          else { //backup match information: maybe we can recover it just after this mismatch
            lengthBak = length;
            indexBak = index;
            delta = true; //enter into delta mode - for the remaining bits in this byte length will be 0; we will exit delta mode and enter into recovery mode on bpos==0
          }
          length = 0;
        }
    }

  if (bpos == 0) {

    // recover match after a 1-byte mismatch
    if (isInPreRecoveryMode()) { // just exited delta mode, so we have a backup
      //the match failed 2 bytes ago, we must increase indexBak by 2:
      indexBak++;
      if (lengthBak < 65535) {
        lengthBak++;
      }
      INJECT_SHARED_c1
        if (buf[indexBak] == c1) { // match continues -> recover
          length = lengthBak;
          index = indexBak;
        }
        else { // still mismatch
          lengthBak = indexBak = 0; // purge backup (give up)
        }
    }

    // extend current match
    if (length != 0) {
      index++;
      if (length < 65535) {
        length++;
      }
      if (isInRecoveryMode() && recoveryModePos() >= minlen_rm) { // recovery seems to be successful and stable -> exit recovery mode
        lengthBak = indexBak = 0; // purge backup
      }
    }
    delta = false;
  }
  if constexpr (false) {
    INJECT_SHARED_bpos
      INJECT_SHARED_blockPos
      printf("  pos %d %d  index %d  length %d  lengthBak %d  delta %d\n", blockPos, bpos, index, length, lengthBak, delta ? 1 : 0);
  }

}

void MatchInfo::registerMatch(const uint32_t pos, const uint32_t LEN, const uint32_t LEN1) {
  assert(pos != 0);
  length = LEN - LEN1 + 1; // rebase
  index = pos;
  lengthBak = indexBak = 0;
  expectedByte = 0;
  delta = false;
}