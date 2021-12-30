#include "DecAlpha.hpp"
#include "../Utils.hpp"
#include <algorithm>
#include <iterator>

std::uint32_t const DECAlpha::op10[] = { 0x40u,0x00u,0x02u,0x49u,0x09u,0x0Bu,0x0Fu,0x12u,0x1Bu,0x1Du,0x60u,0x20u,0x22u,0x69u,0x29u,0x2Bu,0x2Du,0x32u,0x3Bu,0x3Du,0x4Du,0x6Du };
std::uint32_t const DECAlpha::op11[] = { 0x00u,0x08u,0x14u,0x16u,0x20u,0x24u,0x26u,0x28u,0x40u,0x44u,0x46u,0x48u,0x61u,0x64u,0x66u,0x6Cu };
std::uint32_t const DECAlpha::op12[] = { 0x02u,0x06u,0x0Bu,0x12u,0x16u,0x1Bu,0x22u,0x26u,0x2Bu,0x30u,0x31u,0x32u,0x34u,0x36u,0x39u,0x3Bu,0x3Cu,0x52u,0x57u,0x5Au,0x62u,0x67u,0x6Au,0x72u,0x77u,0x7Au };
std::uint32_t const DECAlpha::op13[] = { 0x40u,0x00u,0x60u,0x20u,0x30u };
std::uint32_t const DECAlpha::op14[] = { 0x004u,0x00Au,0x08Au,0x10Au,0x18Au,0x40Au,0x48Au,0x50Au,0x58Au,0x00Bu,0x04Bu,0x08Bu,0x0CBu,0x10Bu,
                                         0x14Bu,0x18BU,0x1CBu,0x50Bu,0x54Bu,0x58Bu,0x5CBu,0x70Bu,0x74Bu,0x78Bu,0x7CBu,0x014u,0x024u,0x02Au,
                                         0x0AAu,0x12Au,0x1AAu,0x42Au,0x4AAu,0x52Au,0x5AAu,0x02Bu,0x06Bu,0x0ABu,0x0EBu,0x12Bu,0x16Bu,0x1ABu,
                                         0x1EBu,0x52Bu,0x56Bu,0x5ABu,0x5EBu,0x72Bu,0x76Bu,0x7ABu,0x7EBu };
std::uint32_t const DECAlpha::op15a[] = { 0x0A5u,0x4A5u,0x0A6u,0x4A6u,0x0A7u,0x4A7u,0x03Cu,0x0BCu,0x03Eu,0x0BEu };
std::uint32_t const DECAlpha::op15b[] = { 0x000u,0x001u,0x002u,0x003u,0x01Eu,0x020u,0x021u,0x022u,0x023u,0x02Cu,0x02Du,0x02Fu };
std::uint32_t const DECAlpha::op16a[] = { 0x0A4u,0x5A4u,0x0A5u,0x5A5u,0x0A6u,0x5A6u,0x0A7u,0x5A7u,0x2ACu,0x6ACu };
std::uint32_t const DECAlpha::op16b[] = { 0x00u,0x01u,0x02u,0x03u,0x20u,0x21u,0x22u,0x23u,0x2Cu,0x2Fu };
std::uint32_t const DECAlpha::op17[] = { 0x010u,0x020u,0x021u,0x022u,0x024u,0x025u,0x02Au,0x02Bu,0x02Cu,0x02Du,0x02Eu,0x02Fu,0x030u,0x130u,0x530u };
std::uint32_t const DECAlpha::op18[] = { 0x0000u,0x0400u,0x4000u,0x4400u,0x8000u,0xA000u,0xC000u,0xE000u,0xE800u,0xF000u,0xF800u,0xFC00u };
std::uint32_t const DECAlpha::op1C[] = { 0x00u,0x01u,0x30u,0x31u,0x32u,0x33u,0x34u,0x35u,0x36u,0x37u,0x38u,0x39u,0x3Au,0x3Bu,0x3Cu,0x3Du,0x3Eu,0x3Fu,0x70u,0x78u };

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
std::uint8_t const DECAlpha::rel_op[] = {
  0x00u, 0x00u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, // 00..07
  0x00u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, // 08..0F
  0x00u, 0x01u, 0x02u, 0x03u, 0x00u, 0x01u, 0x02u, 0x03u, // 10..17
  0x00u, 0x01u, 0x00u, 0x02u, 0x04u, 0x03u, 0x04u, 0x05u, // 18..1F
  0x08u, 0x09u, 0x0Au, 0x0Bu, 0x0Cu, 0x0Du, 0x0Eu, 0x0Fu, // 20..27
  0x10u, 0x11u, 0x12u, 0x13u, 0x14u, 0x15u, 0x16u, 0x17u, // 28..2F
  0x00u, 0x01u, 0x02u, 0x03u, 0x01u, 0x04u, 0x05u, 0x06u, // 30..37
  0x07u, 0x08u, 0x09u, 0x0Au, 0x0Bu, 0x0Cu, 0x0Du, 0x0Eu  // 38..3F
};

bool DECAlpha::IsValidInstruction(std::uint32_t const instruction) {
  std::uint32_t opcode = instruction >> 26u, function = 0u;
  switch (opcode) {
    case 0: { // PAL
      function = instruction & 0x1FFFFFFFu;
      return !(((function > 0x3Fu) && (function < 0x80u)) || (function > 0xBFu));
    }
    case 0x10u: { // INTA
      function = (instruction >> 5u) & 0x7Fu;
      return std::find(std::begin(op10), std::end(op10), function) != std::end(op10);
    }
    case 0x11u: { // INTL
      function = (instruction >> 5u) & 0x7Fu;
      return std::find(std::begin(op11), std::end(op11), function) != std::end(op11);
    }
    case 0x12u: { // INTS
      function = (instruction >> 5u) & 0x7Fu;
      return std::find(std::begin(op12), std::end(op12), function) != std::end(op12);
    }
    case 0x13u: { // INTM
      function = (instruction >> 5u) & 0x7Fu;
      return std::find(std::begin(op13), std::end(op13), function) != std::end(op13);
    }
    case 0x14u: { // ITFP
      function = (instruction >> 5u) & 0x7FFu;
      return std::find(std::begin(op14), std::end(op14), function) != std::end(op14);
    }
    case 0x15u: { // FLTV
      function = (instruction >> 5u) & 0x7FFu;
      if (std::find(std::begin(op15a), std::end(op15a), function) != std::end(op15a))
        return true;
      if ((function & 0x200u) != 0u)
        return false;
      function &= 0x7Fu;
      return std::find(std::begin(op15b), std::end(op15b), function) != std::end(op15b);
    }
    case 0x16u: { // FLTI
      function = (instruction >> 5u) & 0x7FFu;
      if (std::find(std::begin(op16a), std::end(op16a), function) != std::end(op16a))
        return true;
      if (((function & 0x600u) == 0x200u) || ((function & 0x500u) == 0x400u))
        return false;
      opcode = function & 0x3Fu;
      if (std::find(std::begin(op16b), std::end(op16b), opcode) != std::end(op16b))
        return true;
      if ((opcode == 0x3Cu) || (opcode == 0x3Eu))
        return (function & 0x300u) != 0x100u;
      return false;
    }
    case 0x17u: { // FLTL
      function = (instruction >> 5u) & 0x7FFu;
      return std::find(std::begin(op17), std::end(op17), function) != std::end(op17);
    }
    case 0x18u: {
      function = instruction & 0xFFFFu;
      return std::find(std::begin(op18), std::end(op18), function) != std::end(op18);
    }
    case 0x1Cu: { // FPTI
      function = (instruction >> 5u) & 0x7Fu;
      return std::find(std::begin(op1C), std::end(op1C), function) != std::end(op1C);
    }
    default: {
      if ((opcode >= 1u) && (opcode <= 7u))
        return false;
      if (((opcode >= 0x08u) && (opcode <= 0x0Fu)) || ((opcode >= 0x19u) && (opcode <= 0x3Fu)))
        return true;
    }
  }
  return false;
}

void DECAlpha::Shuffle(std::uint32_t& instruction) {
  std::uint32_t const opcode = instruction >> 26u;
  switch (DECAlpha::formats[opcode]) {
    case DECAlpha::F_P: { // op, ra, rb, func, rc => op, func, ra, rb, rc
      instruction = (instruction & 0xFC000000u) | 
                    ((instruction & 0xFFE0u) << 10u) |
                    ((instruction >> 11u) & 0x7FE0u) |
                    (instruction & 0x1Fu);
      break;
    }
    case DECAlpha::Mfc: { // op, ra, rb, func => op, func, ra, rb
      instruction = (instruction & 0xFC000000u) |
                    ((instruction & 0xFFFFu) << 10u) |
                    ((instruction >> 16u) & 0x3FFu);
      break;
    }
    case DECAlpha::Opr: { // op, ra, {(rb, unused), (literal)}, bit, func, rc => op, bit, func, ra, {(rb, unused), (literal)}, rc
      instruction = (instruction & 0xFC000000u) |
                    ((instruction << 13u) & 0x3FC0000u) |
                    ((instruction >> 8u) & 0x3FFE0u) |
                    (instruction & 0x1Fu);
      break;
    }
  }
  instruction = bswap(instruction);
}

void DECAlpha::Unshuffle(std::uint32_t& instruction) {
  instruction = bswap(instruction);
  std::uint32_t const opcode = instruction >> 26u;
  switch (DECAlpha::formats[opcode]) {
    case DECAlpha::F_P: { // op, func, ra, rb, rc => op, ra, rb, func, rc
      instruction = (instruction & 0xFC000000u) |
                    ((instruction >> 10u) & 0xFFE0u) |
                    ((instruction & 0x7FE0u) << 11u) |
                    (instruction & 0x1Fu);
      break;
    }
    case DECAlpha::Mfc: { // op, func, ra, rb => op, ra, rb, func
      instruction = (instruction & 0xFC000000u) |
                    ((instruction >> 10u) & 0xFFFFu) |
                    ((instruction & 0x3FFu) << 16u);
      break;
    }
    case DECAlpha::Opr: { // op, bit, func, ra, {(rb, unused), (literal)}, rc => op, ra, {(rb, unused), (literal)}, bit, func, rc
      instruction = (instruction & 0xFC000000u) |
                    ((instruction & 0x3FC0000u) >> 13u) |
                    ((instruction & 0x3FFE0u) << 8u) |
                    (instruction & 0x1Fu);
      break;
    }
  }
}
