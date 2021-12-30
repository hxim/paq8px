#pragma once

#include <cstdint>

class DECAlpha {
public:
  enum InstructionFormat { Bra, F_P, Mem, Mfc, Mbr, Opr, Pcd, Nop };
  static std::uint32_t const op10[];
  static std::uint32_t const op11[];
  static std::uint32_t const op12[];
  static std::uint32_t const op13[];
  static std::uint32_t const op14[];
  static std::uint32_t const op15a[];
  static std::uint32_t const op15b[];
  static std::uint32_t const op16a[];
  static std::uint32_t const op16b[];
  static std::uint32_t const op17[];
  static std::uint32_t const op18[];
  static std::uint32_t const op1C[];
  static InstructionFormat const formats[];
  static std::uint8_t const rel_op[];
  static bool IsValidInstruction(std::uint32_t const instruction);
  static void Shuffle(std::uint32_t& instruction);
  static void Unshuffle(std::uint32_t& instruction);
};
