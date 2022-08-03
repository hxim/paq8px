#pragma once

#include "../RingBuffer.hpp"
#include "../ContextMap2.hpp"
#include "../Hash.hpp"
#include "../IndirectMap.hpp"
#include "../Mixer.hpp"
#include "../Shared.hpp"
#include <cstdint>
#include <cstring>

/**
 * Model for x86/x64 code.
 * Based on the previous paq* exe models and on DisFilter (http://www.farbrausch.de/~fg/code/disfilter/) by Fabian Giesen.
 *
 * Attempts to parse the input stream as x86/x64 instructions, and quantizes them into 32-bit representations that are then
 * used as context to predict the next bits, and also extracts possible sparse contexts at several previous positions that
 * are relevant to parsing (prefixes, opcode, Mod and R/m fields of the ModRM byte, Scale field of the SIB byte)
 */
class ExeModel {
  // formats
  enum InstructionFormat {
      // encoding mode
      fNM = 0x0, // no ModRM
      fAM = 0x1, // no ModRM, "address mode" (jumps or direct addresses)
      fMR = 0x2, // ModRM present
      fMEXTRA = 0x3, // ModRM present, includes extra bits for opcode
      fMODE = 0x3, // bitmask for mode

      // no ModRM: size of immediate operand
      fNI = 0x0, // no immediate
      fBI = 0x4, // byte immediate
      fWI = 0x8, // word immediate
      fDI = 0xc, // dword immediate
      fTYPE = 0xc, // type mask

      // address mode: type of address operand
      fAD = 0x0, // absolute address
      fDA = 0x4, // dword absolute jump target
      fBR = 0x8, // byte relative jump target
      fDR = 0xc, // dword relative jump target

      // others
      fERR = 0xf, // denotes invalid opcodes
  };

  /**< 1 byte opcodes */
  static constexpr uint8_t table1[256] = {
      // 0       1       2       3       4       5       6       7       8       9       a       b       c       d       e       f
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, // 0
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, // 1
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, // 2
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fNM | fBI, fNM | fDI, fNM | fNI, fNM | fNI, // 3

      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI,
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, // 4
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI,
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, // 5
      fNM | fNI, fNM | fNI, fMR | fNI, fMR | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fDI, fMR | fDI, fNM | fBI,
      fMR | fBI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, // 6
      fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR,
      fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, // 7

      fMR | fBI, fMR | fDI, fMR | fBI, fMR | fBI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // 8
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fAM | fDA,
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, // 9
      fAM | fAD, fAM | fAD, fAM | fAD, fAM | fAD, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fBI, fNM | fDI, fNM | fNI,
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, // a
      fNM | fBI, fNM | fBI, fNM | fBI, fNM | fBI, fNM | fBI, fNM | fBI, fNM | fBI, fNM | fBI, fNM | fDI, fNM | fDI, fNM | fDI,
      fNM | fDI, fNM | fDI, fNM | fDI, fNM | fDI, fNM | fDI, // b

      fMR | fBI, fMR | fBI, fNM | fWI, fNM | fNI, fMR | fNI, fMR | fNI, fMR | fBI, fMR | fDI, fNM | fBI, fNM | fNI, fNM | fWI,
      fNM | fNI, fNM | fNI, fNM | fBI, fERR, fNM | fNI, // c
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fNM | fBI, fNM | fBI, fNM | fNI, fNM | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // d
      fAM | fBR, fAM | fBR, fAM | fBR, fAM | fBR, fNM | fBI, fNM | fBI, fNM | fBI, fNM | fBI, fAM | fDR, fAM | fDR, fAM | fAD,
      fAM | fBR, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, // e
      fNM | fNI, fERR, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fMEXTRA, fMEXTRA, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI,
      fNM | fNI, fNM | fNI, fMEXTRA, fMEXTRA, // f
  };

  /**< 2 byte opcodes */
  static constexpr uint8_t table2[256] = {
      // 0       1       2       3       4       5       6       7       8       9       a       b       c       d       e       f
      fERR, fERR, fERR, fERR, fERR, fERR, fNM | fNI, fERR, fNM | fNI, fNM | fNI, fERR, fERR, fERR, fERR, fERR, fERR, // 0
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fERR, fERR, fERR, fERR, fERR,
      fERR, fERR, // 1
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fERR, fERR, fERR, fERR, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, // 2
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fERR, fNM | fNI, fERR, fERR, fERR, fERR, fERR, fERR, fERR,
      fERR, // 3

      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // 4
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // 5
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // 6
      fMR | fBI, fMR | fBI, fMR | fBI, fMR | fBI, fMR | fNI, fMR | fNI, fMR | fNI, fNM | fNI, fERR, fERR, fERR, fERR, fERR, fERR,
      fMR | fNI, fMR | fNI, // 7

      fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR,
      fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, fAM | fDR, // 8
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // 9
      fNM | fNI, fNM | fNI, fNM | fNI, fMR | fNI, fMR | fBI, fMR | fNI, fMR | fNI, fMR | fNI, fERR, fERR, fERR, fMR | fNI, fMR | fBI,
      fMR | fNI, fERR, fMR | fNI, // a
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fERR, fERR, fERR, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, // b

      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fNM | fNI, fNM | fNI, fNM | fNI,
      fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, fNM | fNI, // c
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // d
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // e
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fERR, // f
  };

  /**< 3 byte opcodes 0F38XX */
  static constexpr uint8_t table3_38[256] = {
      // 0       1       2       3       4       5       6       7       8       9       a       b       c       d       e       f
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fERR, fERR, fERR, fERR, // 0
      fMR | fNI, fERR, fERR, fERR, fMR | fNI, fMR | fNI, fERR, fMR | fNI, fERR, fERR, fERR, fERR, fMR | fNI, fMR | fNI, fMR | fNI,
      fERR, // 1
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fERR, fERR, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fERR,
      fERR, fERR, fERR, // 2
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fERR, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI,
      fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // 3
      fMR | fNI, fMR | fNI, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 4
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 5
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 6
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 7
      fMR | fNI, fMR | fNI, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 8
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 9
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // a
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // b
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // c
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // d
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // e
      fMR | fNI, fMR | fNI, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // f
  };

  /**< 3 byte opcodes 0F3AXX */
  static constexpr uint8_t table3_3A[256] = {
      // 0       1       2       3       4       5       6       7       8       9       a       b       c       d       e       f
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fMR | fBI, fMR | fBI, fMR | fBI, fMR | fBI, fMR | fBI, fMR | fBI, fMR | fBI,
      fMR | fBI, // 0
      fERR, fERR, fERR, fERR, fMR | fBI, fMR | fBI, fMR | fBI, fMR | fBI, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 1
      fMR | fBI, fMR | fBI, fMR | fBI, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 2
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 3
      fMR | fBI, fMR | fBI, fMR | fBI, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 4
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 5
      fMR | fBI, fMR | fBI, fMR | fBI, fMR | fBI, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 6
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 7
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 8
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // 9
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // a
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // b
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // c
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // d
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // e
      fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, fERR, // f
  };

  /**< escape opcodes using ModRM byte to get more variants */
  static constexpr uint8_t tableX[32] = {
      // 0       1       2       3       4       5       6       7
      fMR | fBI, fERR, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // escapes for 0xf6
      fMR | fDI, fERR, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, fMR | fNI, // escapes for 0xf7
      fMR | fNI, fMR | fNI, fERR, fERR, fERR, fERR, fERR, fERR, // escapes for 0xfe
      fMR | fNI, fMR | fNI, fMR | fNI, fERR, fMR | fNI, fERR, fMR | fNI, fERR, // escapes for 0xff
  };

  static constexpr uint8_t invalidX64Ops[19] = {0x06, 0x07, 0x16, 0x17, 0x1E, 0x1F, 0x27, 0x2F, 0x37, 0x3F, 0x60, 0x61, 0x62, 0x82, 0x9A, 0xD4, 0xD5, 0xD6, 0xEA,};
  static constexpr uint8_t x64Prefixes[8] = {0x26, 0x2E, 0x36, 0x3E, 0x9B, 0xF0, 0xF2, 0xF3,};

  enum InstructionCategory {
      OP_INVALID = 0,
      OP_PREFIX_SEGREG = 1,
      OP_PREFIX = 2,
      OP_PREFIX_X87FPU = 3,
      OP_GEN_DATAMOV = 4,
      OP_GEN_STACK = 5,
      OP_GEN_CONVERSION = 6,
      OP_GEN_ARITH_DECIMAL = 7,
      OP_GEN_ARITH_BINARY = 8,
      OP_GEN_LOGICAL = 9,
      OP_GEN_SHF_ROT = 10,
      OP_GEN_BIT = 11,
      OP_GEN_BRANCH = 12,
      OP_GEN_BRANCH_COND = 13,
      OP_GEN_BREAK = 14,
      OP_GEN_STRING = 15,
      OP_GEN_INOUT = 16,
      OP_GEN_FLAG_CONTROL = 17,
      OP_GEN_SEGREG = 18,
      OP_GEN_CONTROL = 19,
      OP_SYSTEM = 20,
      OP_X87_DATAMOV = 21,
      OP_X87_ARITH = 22,
      OP_X87_COMPARISON = 23,
      OP_X87_TRANSCENDENTAL = 24,
      OP_X87_LOAD_CONSTANT = 25,
      OP_X87_CONTROL = 26,
      OP_X87_CONVERSION = 27,
      OP_STATE_MANAGEMENT = 28,
      OP_MMX = 29,
      OP_SSE = 30,
      OP_SSE_DATAMOV = 31,
  };

  static constexpr uint8_t typeOp1[256] = {
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //03
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_STACK, OP_GEN_STACK, //07
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, //0B
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_STACK, OP_PREFIX, //0F
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //13
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_STACK, OP_GEN_STACK, //17
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //1B
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_STACK, OP_GEN_STACK, //1F
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, //23
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_PREFIX_SEGREG, OP_GEN_ARITH_DECIMAL, //27
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //2B
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_PREFIX_SEGREG, OP_GEN_ARITH_DECIMAL, //2F
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, //33
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_PREFIX_SEGREG, OP_GEN_ARITH_DECIMAL, //37
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //3B
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_PREFIX_SEGREG, OP_GEN_ARITH_DECIMAL, //3F
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //43
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //47
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //4B
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //4F
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, //53
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, //57
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, //5B
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, OP_GEN_STACK, //5F
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_BREAK, OP_GEN_CONVERSION, //63
    OP_PREFIX_SEGREG, OP_PREFIX_SEGREG, OP_PREFIX, OP_PREFIX, //67
    OP_GEN_STACK, OP_GEN_ARITH_BINARY, OP_GEN_STACK, OP_GEN_ARITH_BINARY, //6B
    OP_GEN_INOUT, OP_GEN_INOUT, OP_GEN_INOUT, OP_GEN_INOUT, //6F
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //73
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //77
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //7B
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //7F
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //83
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //87
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //8B
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_STACK, //8F
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //93
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //97
    OP_GEN_CONVERSION, OP_GEN_CONVERSION, OP_GEN_BRANCH, OP_PREFIX_X87FPU, //9B
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //9F
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //A3
    OP_GEN_STRING, OP_GEN_STRING, OP_GEN_STRING, OP_GEN_STRING, //A7
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_STRING, OP_GEN_STRING, //AB
    OP_GEN_STRING, OP_GEN_STRING, OP_GEN_STRING, OP_GEN_STRING, //AF
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //B3
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //B7
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //BB
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //BF
    OP_GEN_SHF_ROT, OP_GEN_SHF_ROT, OP_GEN_BRANCH, OP_GEN_BRANCH, //C3
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //C7
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_BRANCH, OP_GEN_BRANCH, //CB
    OP_GEN_BREAK, OP_GEN_BREAK, OP_GEN_BREAK, OP_GEN_BREAK, //CF
    OP_GEN_SHF_ROT, OP_GEN_SHF_ROT, OP_GEN_SHF_ROT, OP_GEN_SHF_ROT, //D3
    OP_GEN_ARITH_DECIMAL, OP_GEN_ARITH_DECIMAL, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //D7
    OP_X87_ARITH, OP_X87_DATAMOV, OP_X87_ARITH, OP_X87_DATAMOV, //DB
    OP_X87_ARITH, OP_X87_DATAMOV, OP_X87_ARITH, OP_X87_DATAMOV, //DF
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //E3
    OP_GEN_INOUT, OP_GEN_INOUT, OP_GEN_INOUT, OP_GEN_INOUT, //E7
    OP_GEN_BRANCH, OP_GEN_BRANCH, OP_GEN_BRANCH, OP_GEN_BRANCH, //EB
    OP_GEN_INOUT, OP_GEN_INOUT, OP_GEN_INOUT, OP_GEN_INOUT, //EF
    OP_PREFIX, OP_GEN_BREAK, OP_PREFIX, OP_PREFIX, //F3
    OP_SYSTEM, OP_GEN_FLAG_CONTROL, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, //F7
    OP_GEN_FLAG_CONTROL, OP_GEN_FLAG_CONTROL, OP_GEN_FLAG_CONTROL, OP_GEN_FLAG_CONTROL, //FB
    OP_GEN_FLAG_CONTROL, OP_GEN_FLAG_CONTROL, OP_GEN_ARITH_BINARY, OP_GEN_BRANCH, //FF
  };

  static constexpr uint8_t typeOp2[256] = {
    OP_SYSTEM, OP_SYSTEM, OP_SYSTEM, OP_SYSTEM, //03
    OP_INVALID, OP_SYSTEM, OP_SYSTEM, OP_SYSTEM, //07
    OP_SYSTEM, OP_SYSTEM, OP_INVALID, OP_GEN_CONTROL, //0B
    OP_INVALID, OP_GEN_CONTROL, OP_INVALID, OP_INVALID, //0F
    OP_SSE_DATAMOV, OP_SSE_DATAMOV, OP_SSE_DATAMOV, OP_SSE_DATAMOV, //13
    OP_SSE, OP_SSE, OP_SSE_DATAMOV, OP_SSE_DATAMOV, //17
    OP_SSE, OP_GEN_CONTROL, OP_GEN_CONTROL, OP_GEN_CONTROL, //1B
    OP_GEN_CONTROL, OP_GEN_CONTROL, OP_GEN_CONTROL, OP_GEN_CONTROL, //1F
    OP_SYSTEM, OP_SYSTEM, OP_SYSTEM, OP_SYSTEM, //23
    OP_SYSTEM, OP_INVALID, OP_SYSTEM, OP_INVALID, //27
    OP_SSE_DATAMOV, OP_SSE_DATAMOV, OP_SSE, OP_SSE, //2B
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //2F
    OP_SYSTEM, OP_SYSTEM, OP_SYSTEM, OP_SYSTEM, //33
    OP_SYSTEM, OP_SYSTEM, OP_INVALID, OP_INVALID, //37
    OP_PREFIX, OP_INVALID, OP_PREFIX, OP_INVALID, //3B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //3F
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //43
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //47
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //4B
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //4F
    OP_SSE_DATAMOV, OP_SSE, OP_SSE, OP_SSE, //53
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //57
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //5B
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //5F
    OP_MMX, OP_MMX, OP_MMX, OP_MMX, //63
    OP_MMX, OP_MMX, OP_MMX, OP_MMX, //67
    OP_MMX, OP_MMX, OP_MMX, OP_MMX, //6B
    OP_INVALID, OP_INVALID, OP_MMX, OP_MMX, //6F
    OP_SSE, OP_MMX, OP_MMX, OP_MMX, //73
    OP_MMX, OP_MMX, OP_MMX, OP_MMX, //77
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //7B
    OP_INVALID, OP_INVALID, OP_MMX, OP_MMX, //7F
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //83
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //87
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //8B
    OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, OP_GEN_BRANCH_COND, //8F
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //93
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //97
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //9B
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //9F
    OP_GEN_STACK, OP_GEN_STACK, OP_GEN_CONTROL, OP_GEN_BIT, //A3
    OP_GEN_SHF_ROT, OP_GEN_SHF_ROT, OP_INVALID, OP_INVALID, //A7
    OP_GEN_STACK, OP_GEN_STACK, OP_SYSTEM, OP_GEN_BIT, //AB
    OP_GEN_SHF_ROT, OP_GEN_SHF_ROT, OP_STATE_MANAGEMENT, OP_GEN_ARITH_BINARY, //AF
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_BIT, //B3
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_CONVERSION, OP_GEN_CONVERSION, //B7
    OP_INVALID, OP_GEN_CONTROL, OP_GEN_BIT, OP_GEN_BIT, //BB
    OP_GEN_BIT, OP_GEN_BIT, OP_GEN_CONVERSION, OP_GEN_CONVERSION, //BF
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_SSE, OP_SSE, //C3
    OP_SSE, OP_SSE, OP_SSE, OP_GEN_DATAMOV, //C7
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //CB
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_GEN_DATAMOV, //CF
    OP_INVALID, OP_MMX, OP_MMX, OP_MMX, //D3
    OP_SSE, OP_MMX, OP_INVALID, OP_SSE, //D7
    OP_MMX, OP_MMX, OP_SSE, OP_MMX, //DB
    OP_MMX, OP_MMX, OP_SSE, OP_MMX, //DF
    OP_SSE, OP_MMX, OP_SSE, OP_MMX, //E3
    OP_SSE, OP_MMX, OP_INVALID, OP_SSE, //E7
    OP_MMX, OP_MMX, OP_SSE, OP_MMX, //EB
    OP_MMX, OP_MMX, OP_SSE, OP_MMX, //EF
    OP_INVALID, OP_MMX, OP_MMX, OP_MMX, //F3
    OP_SSE, OP_MMX, OP_SSE, OP_SSE, //F7
    OP_MMX, OP_MMX, OP_MMX, OP_SSE, //FB
    OP_MMX, OP_MMX, OP_MMX, OP_INVALID, //FF
  };

  static constexpr uint8_t typeOp3_38[256] = {
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //03
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //07
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //0B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //0F
    OP_SSE, OP_INVALID, OP_INVALID, OP_INVALID, //13
    OP_SSE, OP_SSE, OP_INVALID, OP_SSE, //17
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //1B
    OP_SSE, OP_SSE, OP_SSE, OP_INVALID, //1F
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //23
    OP_SSE, OP_SSE, OP_INVALID, OP_INVALID, //27
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //2B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //2F
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //33
    OP_SSE, OP_SSE, OP_INVALID, OP_SSE, //37
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //3B
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //3F
    OP_SSE, OP_SSE, OP_INVALID, OP_INVALID, //43
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //47
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //4B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //4F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //53
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //57
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //5B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //5F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //63
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //67
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //6B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //6F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //73
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //77
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //7B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //7F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //83
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //87
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //8B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //8F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //93
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //97
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //9B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //9F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //A3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //A7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //AB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //AF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //B3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //B7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //BB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //BF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //C3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //C7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //CB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //CF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //D3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //D7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //DB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //DF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //E3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //E7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //EB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //EF
    OP_GEN_DATAMOV, OP_GEN_DATAMOV, OP_INVALID, OP_INVALID, //F3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //F7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //FB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //FF
  };

  static constexpr uint8_t typeOp3_3A[256] = {
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //03
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //07
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //0B
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //0F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //13
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //17
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //1B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //1F
    OP_SSE, OP_SSE, OP_SSE, OP_INVALID, //23
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //27
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //2B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //2F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //33
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //37
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //3B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //3F
    OP_SSE, OP_SSE, OP_SSE, OP_INVALID, //43
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //47
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //4B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //4F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //53
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //57
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //5B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //5F
    OP_SSE, OP_SSE, OP_SSE, OP_SSE, //63
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //67
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //6B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //6F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //73
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //77
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //7B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //7F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //83
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //87
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //8B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //8F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //93
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //97
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //9B
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //9F
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //A3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //A7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //AB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //AF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //B3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //B7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //BB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //BF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //C3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //C7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //CB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //CF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //D3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //D7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //DB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //DF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //E3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //E7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //EB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //EF
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //F3
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //F7
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //FB
    OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, //FF
  };

  static constexpr uint8_t typeOpX[32] = {
    // escapes for F6
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY,
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY,
    // escapes for F7
    OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_LOGICAL, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY,
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY,
    // escapes for FE
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID, OP_INVALID,
    // escapes for FF
    OP_GEN_ARITH_BINARY, OP_GEN_ARITH_BINARY, OP_GEN_BRANCH, OP_GEN_BRANCH, OP_GEN_BRANCH, OP_GEN_BRANCH, OP_GEN_STACK,
    OP_INVALID,
  };

  enum Prefixes {
    ES_OVERRIDE = 0x26,
    CS_OVERRIDE = 0x2E,
    SS_OVERRIDE = 0x36,
    DS_OVERRIDE = 0x3E,
    FS_OVERRIDE = 0x64,
    GS_OVERRIDE = 0x65,
    AD_OVERRIDE = 0x67,
    WAIT_FPU = 0x9B,
    LOCK = 0xF0,
    REP_N_STR = 0xF2,
    REP_STR = 0xF3,
  };

  /**
    * 1-byte opcodes of special interest (for one reason or another)
    */
  enum Opcodes {
    OP_2BYTE = 0x0f, /**< start of 2-byte opcode */
    OP_OSIZE = 0x66, /**< operand size prefix */
    OP_CALLF = 0x9a, OP_RETNI = 0xc2, /**< ret near+immediate */
    OP_RETN = 0xc3, OP_ENTER = 0xc8, OP_INT3 = 0xcc, OP_INTO = 0xce, OP_CALLN = 0xe8, OP_JMPF = 0xea, OP_ICEBP = 0xf1,
  };

  enum ExeState {
    Start = 0,
    PrefOpSize = 1,
    PrefMultiByteOp = 2,
    ParseFlags = 3,
    ExtraFlags = 4,
    ReadModRM = 5,
    Read_OP3_38 = 6,
    Read_OP3_3A = 7,
    ReadSIB = 8,
    Read8 = 9,
    Read16 = 10,
    Read32 = 11,
    Read8ModRm = 12,
    Read16F = 13,
    Read32ModRm = 14,
    Error = 15,
  };

  static constexpr uint32_t CacheSize = 32;

  struct OpCache {
      uint32_t Op[CacheSize];
      uint32_t Index;
  };

  struct Instruction {
      uint32_t data;
      uint8_t prefix, code, ModRM, SIB, REX, flags, bytesRead, size, category;
      bool mustCheckRex, decoding, o16, imm8;
  };

  static constexpr int codeShift = 3;
  static constexpr uint32_t codeMask = 0xFF << codeShift; /**< 0x000007F8 */
  static constexpr uint32_t clearCodeMask = 0xFFFFFFFF ^ codeMask; /**< 0xFFFFF807 */
  static constexpr uint32_t prefixMask = (1 << codeShift) - 1; /**< 0x07 */
  static constexpr uint32_t operandSizeOverride = 0x01 << (8 + codeShift); /**< 0x00000800 */
  static constexpr uint32_t multiByteOpcode = 0x02 << (8 + codeShift); /**< 0x00001000 */
  static constexpr uint32_t prefixRex = 0x04 << (8 + codeShift); /**< 0x00002000 */
  static constexpr uint32_t prefix38 = 0x08 << (8 + codeShift); /**< 0x00004000 */
  static constexpr uint32_t prefix3A = 0x10 << (8 + codeShift); /**< 0x00008000 */
  static constexpr uint32_t hasExtraFlags = 0x20 << (8 + codeShift); /**< 0x00010000 */
  static constexpr uint32_t hasModRm = 0x40 << (8 + codeShift); /**< 0x00020000 */
  static constexpr int ModRMShift = 7 + 8 + codeShift; /**< 18 */
  static constexpr int SIBScaleShift = ModRMShift + 8 - 6; /**< 20 */
  static constexpr uint32_t regDWordDisplacement = 1 << (8 + SIBScaleShift); /**< 0x10000000 */
  static constexpr uint32_t addressMode = 2 << (8 + SIBScaleShift); /**< 0x20000000 */
  static constexpr int typeShift = 2 + 8 + SIBScaleShift; /**< 30 */
  static constexpr uint32_t categoryShift = 5;
  static constexpr uint32_t categoryMask = ((1 << categoryShift) - 1); /**< 0x1F (31) */
  static constexpr uint8_t ModRM_mod = 0xC0;
  static constexpr uint8_t ModRM_reg = 0x38;
  static constexpr uint8_t ModRM_rm = 0x07;
  static constexpr uint8_t SIB_scale = 0xC0;
  static constexpr uint8_t SIB_index = 0x38; /**< unused */
  static constexpr uint8_t SIB_base = 0x07;
  static constexpr uint8_t REX_w = 0x08;
  static constexpr uint32_t minRequired = 8; /**< minimum required consecutive valid instructions to be considered as code */
private:
  static constexpr int nCM1 = 10, nCM2 = 10, nIM = 1;
  Shared * const shared;
  ContextMap2 cm;
  IndirectMap iMap;
  OpCache cache {};
  uint32_t stateBh[256] {};
  ExeState pState, state;
  Instruction op {};
  uint32_t totalOps, opMask, opCategoryMask, context;
  uint64_t brkCtx; /**< hash */
  bool valid;

  static inline bool isInvalidX64Op(uint8_t op);
  static inline bool isValidX64Prefix(uint8_t prefix);
  static void processMode(Instruction &op, ExeState &state);
  static void processFlags2(Instruction &op, ExeState &state);
  static void processFlags(Instruction &op, ExeState &state);
  static void checkFlags(Instruction &op, ExeState &state);
  static void readFlags(Instruction &op, ExeState &state);
  static void processModRm(Instruction &op, ExeState &state);
  static void applyCodeAndSetFlag(Instruction &op, uint32_t flag = 0);
  static inline uint32_t opN(OpCache &cache, uint32_t n);
  static inline uint32_t opNCategory(uint32_t &mask, uint32_t n);
  inline uint32_t pref(uint32_t i);

  /**
    * Get context at buf(i) relevant to parsing 32-bit x86 code.
    * @param i
    * @param x
    * @return
    */
  uint32_t exeCxt(uint32_t i, uint32_t x);

  void update();

public:
  static constexpr int MIXERINPUTS =
          (nCM1 + nCM2) * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY) +
          nIM * IndirectMap::MIXERINPUTS; /**< 142 */
  static constexpr int MIXERCONTEXTS = 1024 + 1024 + 1024 + 8192 + 8192 + 8192; /**< 27648 */
  static constexpr int MIXERCONTEXTSETS = 6;

  ExeModel(Shared* const sh, const uint64_t size) : shared(sh), 
          cm(sh, size, nCM1 + nCM2, 64), iMap(sh, 20, 1, 64, 1023), pState(Start), state(Start),
          totalOps(0), opMask(0), opCategoryMask(0), context(0), brkCtx(0), valid(false) {
    assert(isPowerOf2(size));
    memset(&cache, 0, sizeof(OpCache));
    memset(&op, 0, sizeof(Instruction));
    memset(&stateBh, 0, sizeof(stateBh));
  }

  void mix(Mixer &m);
};
