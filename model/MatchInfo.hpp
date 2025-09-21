#pragma once

#include "../Shared.hpp"
#include "../ContextMap2.hpp"
#include "../HashElementForMatchPositions.hpp"
#include "../IndirectContext.hpp"
#include "../LargeStationaryMap.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../StationaryMap.hpp"

struct MatchInfo {
  uint32_t length = 0;
  uint32_t index = 0;
  uint32_t lengthBak = 0;
  uint32_t indexBak = 0;
  uint8_t expectedByte = 0;
  bool delta = false;

  bool isInNoMatchMode() const;
  bool isInPreRecoveryMode() const;
  bool isInRecoveryMode() const;
  uint32_t recoveryModePos() const;
  uint64_t prio() const;
  bool isBetterThan(const MatchInfo* other) const;

  void update(Shared* shared, uint32_t minlen_rm);
  void registerMatch(uint32_t pos, uint32_t LEN, uint32_t LEN1);
};
