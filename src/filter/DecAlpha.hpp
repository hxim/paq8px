#pragma once

#include <cstdint>

class DECAlpha {
public:
  enum InstructionFormat { Bra, F_P, Mem, Mfc, Mbr, Opr, Pcd, Nop };
  static uint32_t const op10[];
  static uint32_t const op11[];
  static uint32_t const op12[];
  static uint32_t const op13[];
  static uint32_t const op14[];
  static uint32_t const op15a[];
  static uint32_t const op15b[];
  static uint32_t const op16a[];
  static uint32_t const op16b[];
  static uint32_t const op17[];
  static uint32_t const op18[];
  static uint32_t const op1C[];
  static InstructionFormat const formats[];
  static uint8_t const rel_op[];
  static bool IsValidInstruction(uint32_t const instruction);
  static void Shuffle(uint32_t& instruction);
  static void Unshuffle(uint32_t& instruction);
};
