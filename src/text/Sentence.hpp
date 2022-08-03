#pragma once

#include "Segment.hpp"

class Sentence : public Segment {
public:
  enum Types { // possible sentence types, excluding Imperative
      Declarative, Interrogative, Exclamative, Count
  };
  Types type;
  uint32_t segmentCount {};
  uint32_t verbIndex {}; /**< relative position of last detected verb */
  uint32_t nounIndex {}; /**< relative position of last detected noun */
  uint32_t capitalIndex {}; /**< relative position of last capitalized word, excluding the initial word of this sentence */
  Word lastVerb;
  Word lastNoun;
  Word lastCapital;
};
