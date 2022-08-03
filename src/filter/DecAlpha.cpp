#include "DecAlpha.hpp"
#include "../Utils.hpp"
#include <algorithm>
#include <iterator>

uint32_t const DECAlpha::op10[] = { 0x40,0x00,0x02,0x49,0x09,0x0B,0x0F,0x12,0x1B,0x1D,0x60,0x20,0x22,0x69,0x29,0x2B,0x2D,0x32,0x3B,0x3D,0x4D,0x6D };
uint32_t const DECAlpha::op11[] = { 0x00,0x08,0x14,0x16,0x20,0x24,0x26,0x28,0x40,0x44,0x46,0x48,0x61,0x64,0x66,0x6C };
uint32_t const DECAlpha::op12[] = { 0x02,0x06,0x0B,0x12,0x16,0x1B,0x22,0x26,0x2B,0x30,0x31,0x32,0x34,0x36,0x39,0x3B,0x3C,0x52,0x57,0x5A,0x62,0x67,0x6A,0x72,0x77,0x7A };
uint32_t const DECAlpha::op13[] = { 0x40,0x00,0x60,0x20,0x30 };
uint32_t const DECAlpha::op14[] = { 0x004,0x00A,0x08A,0x10A,0x18A,0x40A,0x48A,0x50A,0x58A,0x00B,0x04B,0x08B,0x0CB,0x10B,
                                         0x14B,0x18B,0x1CB,0x50B,0x54B,0x58B,0x5CB,0x70B,0x74B,0x78B,0x7CB,0x014,0x024,0x02A,
                                         0x0AA,0x12A,0x1AA,0x42A,0x4AA,0x52A,0x5AA,0x02B,0x06B,0x0AB,0x0EB,0x12B,0x16B,0x1AB,
                                         0x1EB,0x52B,0x56B,0x5AB,0x5EB,0x72B,0x76B,0x7AB,0x7EB };
uint32_t const DECAlpha::op15a[] = { 0x0A5,0x4A5,0x0A6,0x4A6,0x0A7,0x4A7,0x03C,0x0BC,0x03E,0x0BE };
uint32_t const DECAlpha::op15b[] = { 0x000,0x001,0x002,0x003,0x01E,0x020,0x021,0x022,0x023,0x02C,0x02D,0x02F };
uint32_t const DECAlpha::op16a[] = { 0x0A4,0x5A4,0x0A5,0x5A5,0x0A6,0x5A6,0x0A7,0x5A7,0x2AC,0x6AC };
uint32_t const DECAlpha::op16b[] = { 0x00,0x01,0x02,0x03,0x20,0x21,0x22,0x23,0x2C,0x2F };
uint32_t const DECAlpha::op17[] = { 0x010,0x020,0x021,0x022,0x024,0x025,0x02A,0x02B,0x02C,0x02D,0x02E,0x02F,0x030,0x130,0x530 };
uint32_t const DECAlpha::op18[] = { 0x0000,0x0400,0x4000,0x4400,0x8000,0xA000,0xC000,0xE000,0xE800,0xF000,0xF800,0xFC00 };
uint32_t const DECAlpha::op1C[] = { 0x00,0x01,0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x3B,0x3C,0x3D,0x3E,0x3F,0x70,0x78 };

DECAlpha::InstructionFormat const DECAlpha::formats[] = {
  DECAlpha::Pcd, DECAlpha::Nop, DECAlpha::Nop, DECAlpha::Nop, DECAlpha::Nop, DECAlpha::Nop, DECAlpha::Nop, DECAlpha::Nop, // 00..07
  DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, // 08..0F
  DECAlpha::Opr, DECAlpha::Opr, DECAlpha::Opr, DECAlpha::Opr, DECAlpha::F_P, DECAlpha::F_P, DECAlpha::F_P, DECAlpha::F_P, // 10..17
  DECAlpha::Mfc, DECAlpha::Pcd, DECAlpha::Mbr, DECAlpha::Pcd, DECAlpha::Opr, DECAlpha::Pcd, DECAlpha::Pcd, DECAlpha::Pcd, // 18..1F
  DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, // 20..27
  DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, DECAlpha::Mem, // 28..2F
  DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Mbr, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, // 30..37
  DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra, DECAlpha::Bra  // 38..3F
};

/*
Bra: 15 opcodes
F_P:  4 opcodes, 108 functions
Mem: 24 opcodes
Mfc:  1 opcode
Mbr:  2 opcodes
Opr:  5 opcodes,  89 functions
Pcd:  6 opcodes
Nop:  7 opcodes
*/
uint8_t const DECAlpha::rel_op[] = {
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, // 00..07
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // 08..0F
  0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, // 10..17
  0x00, 0x01, 0x00, 0x02, 0x04, 0x03, 0x04, 0x05, // 18..1F
  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, // 20..27
  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, // 28..2F
  0x00, 0x01, 0x02, 0x03, 0x01, 0x04, 0x05, 0x06, // 30..37
  0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E  // 38..3F
};

bool DECAlpha::IsValidInstruction(uint32_t const instruction) {
  uint32_t opcode = instruction >> 26, function = 0;
  switch (opcode) {
    case 0: { // PAL
      function = instruction & 0x1FFFFFFF;
      return !(((function > 0x3F) && (function < 0x80)) || (function > 0xBF));
    }
    case 0x10: { // INTA
      function = (instruction >> 5) & 0x7F;
      return std::find(std::begin(op10), std::end(op10), function) != std::end(op10);
    }
    case 0x11: { // INTL
      function = (instruction >> 5) & 0x7F;
      return std::find(std::begin(op11), std::end(op11), function) != std::end(op11);
    }
    case 0x12: { // INTS
      function = (instruction >> 5) & 0x7F;
      return std::find(std::begin(op12), std::end(op12), function) != std::end(op12);
    }
    case 0x13: { // INTM
      function = (instruction >> 5) & 0x7F;
      return std::find(std::begin(op13), std::end(op13), function) != std::end(op13);
    }
    case 0x14: { // ITFP
      function = (instruction >> 5) & 0x7FF;
      return std::find(std::begin(op14), std::end(op14), function) != std::end(op14);
    }
    case 0x15: { // FLTV
      function = (instruction >> 5) & 0x7FF;
      if (std::find(std::begin(op15a), std::end(op15a), function) != std::end(op15a))
        return true;
      if ((function & 0x200) != 0)
        return false;
      function &= 0x7F;
      return std::find(std::begin(op15b), std::end(op15b), function) != std::end(op15b);
    }
    case 0x16: { // FLTI
      function = (instruction >> 5) & 0x7FF;
      if (std::find(std::begin(op16a), std::end(op16a), function) != std::end(op16a))
        return true;
      if (((function & 0x600) == 0x200) || ((function & 0x500) == 0x400))
        return false;
      opcode = function & 0x3F;
      if (std::find(std::begin(op16b), std::end(op16b), opcode) != std::end(op16b))
        return true;
      if ((opcode == 0x3C) || (opcode == 0x3E))
        return (function & 0x300) != 0x100;
      return false;
    }
    case 0x17: { // FLTL
      function = (instruction >> 5) & 0x7FF;
      return std::find(std::begin(op17), std::end(op17), function) != std::end(op17);
    }
    case 0x18: {
      function = instruction & 0xFFFF;
      return std::find(std::begin(op18), std::end(op18), function) != std::end(op18);
    }
    case 0x1C: { // FPTI
      function = (instruction >> 5) & 0x7F;
      return std::find(std::begin(op1C), std::end(op1C), function) != std::end(op1C);
    }
    default: {
      if ((opcode >= 1) && (opcode <= 7))
        return false;
      if (((opcode >= 0x08) && (opcode <= 0x0F)) || ((opcode >= 0x19) && (opcode <= 0x3F)))
        return true;
    }
  }
  return false;
}

void DECAlpha::Shuffle(uint32_t& instruction) {
  uint32_t const opcode = instruction >> 26;
  switch (DECAlpha::formats[opcode]) {
    case DECAlpha::F_P: { // op, ra, rb, func, rc => op, func, ra, rb, rc
      instruction = (instruction & 0xFC000000) | 
                    ((instruction & 0xFFE0) << 10) |
                    ((instruction >> 11) & 0x7FE0) |
                    (instruction & 0x1F);
      break;
    }
    case DECAlpha::Mfc: { // op, ra, rb, func => op, func, ra, rb
      instruction = (instruction & 0xFC000000) |
                    ((instruction & 0xFFFF) << 10) |
                    ((instruction >> 16) & 0x3FF);
      break;
    }
    case DECAlpha::Opr: { // op, ra, {(rb, unused), (literal)}, bit, func, rc => op, bit, func, ra, {(rb, unused), (literal)}, rc
      instruction = (instruction & 0xFC000000) |
                    ((instruction << 13) & 0x3FC0000) |
                    ((instruction >> 8) & 0x3FFE0) |
                    (instruction & 0x1F);
      break;
    }
  }
  instruction = bswap(instruction);
}

void DECAlpha::Unshuffle(uint32_t& instruction) {
  instruction = bswap(instruction);
  uint32_t const opcode = instruction >> 26;
  switch (DECAlpha::formats[opcode]) {
    case DECAlpha::F_P: { // op, func, ra, rb, rc => op, ra, rb, func, rc
      instruction = (instruction & 0xFC000000) |
                    ((instruction >> 10) & 0xFFE0) |
                    ((instruction & 0x7FE0) << 11) |
                    (instruction & 0x1F);
      break;
    }
    case DECAlpha::Mfc: { // op, func, ra, rb => op, ra, rb, func
      instruction = (instruction & 0xFC000000) |
                    ((instruction >> 10) & 0xFFFF) |
                    ((instruction & 0x3FF) << 16);
      break;
    }
    case DECAlpha::Opr: { // op, bit, func, ra, {(rb, unused), (literal)}, rc => op, ra, {(rb, unused), (literal)}, bit, func, rc
      instruction = (instruction & 0xFC000000) |
                    ((instruction & 0x3FC0000) >> 13) |
                    ((instruction & 0x3FFE0) << 8) |
                    (instruction & 0x1F);
      break;
    }
  }
}
