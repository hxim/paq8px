#pragma once

#include "../filter/DecAlpha.hpp"
#include "../Shared.hpp"
#include "../Mixer.hpp"
#include "../RingBuffer.hpp"
#include "../IndirectMap.hpp"

class DECAlphaModel {
public:
  enum State {
    OpCode,
    Bra_Ra,
    Bra_Displacement,
    F_P_Function,
    F_P_Ra,
    F_P_Rb,
    F_P_Rc,
    Mem_Ra,
    Mem_Rb,
    Mem_Displacement,
    Mfc_Function,
    Mfc_Ra,
    Mfc_Rb,
    Mbr_Ra,
    Mbr_Rb,
    Mbr_Displacement,
    Opr_Bit,
    Opr_Function,
    Opr_Ra,
    Opr_Rb,
    Opr_Unused,
    Opr_Literal,
    Opr_Rc,
    Pcd_Function,
    Nop_Skip,
    Count
  };
  struct Instruction {
    uint32_t Function, Displacement;
    uint8_t Opcode, Bit, Ra, Rb, Rc, Literal, relOpcode, Format;
  };
private:
  static constexpr uint32_t nMaps = 11;
  static constexpr uint32_t maps_mask[nMaps-1] = { 0x6FC3FF, 0x6F0387, 0x4E0383, 0x440183, 0x181, 0x181, 0x81, 0x1, 0x1, 0x1 };
  Shared* const shared;
  RingBuffer<Instruction> cache;
  State state;
  Instruction op;
  uint32_t count;
  uint8_t lastRc;
  Instruction last[8];
  IndirectMap maps0[State::Count], maps1[18], maps2[12], maps3[9], maps4[6], maps5[3], maps6[3], maps7[2], maps8[1], maps9[1], maps10[1];
  IndirectMap* const maps[nMaps - 1] = { &maps1[0], &maps2[0], &maps3[0], &maps4[0], &maps5[0], &maps6[0], &maps7[0], &maps8[0], &maps9[0], &maps10[0] };
  constexpr int32_t map_state(uint32_t const map, State const state) {
    assert(map < 10);
    int32_t r = -1;
    if (((maps_mask[map] >> state) & 1) != 0)     
      for (int32_t i = state; i >= 0; r += (maps_mask[map] >> i) & 1, i--);
    return r;
  }
public:
  static constexpr int MIXERINPUTS = nMaps * IndirectMap::MIXERINPUTS;
  static constexpr int MIXERCONTEXTS = State::Count * 26 + State::Count * 64 + 2048 + 4096 + 4096 + 8192 + 8192 + 8192 + 4096 + 4096 + 4096;
  static constexpr int MIXERCONTEXTSETS = 11;
  DECAlphaModel(Shared* const sh) : 
    shared(sh), cache(8), op{0}, state(OpCode), count(0), lastRc(0xFF), last{0},
    maps0 {
      {sh, 1, 6, 256, 255}, // OpCode
      {sh, 1, 5, 256, 255}, // Bra_Ra
      {sh, 2, 7, 256, 255}, // Bra_Displacement
      {sh, 2, 3, 256, 255}, // F_P_Function
      {sh, 1, 5, 256, 255}, // F_P_Ra
      {sh, 1, 5, 256, 255}, // F_P_Rb
      {sh, 1, 5, 256, 255}, // F_P_Rc
      {sh, 1, 5, 256, 255}, // Mem_Ra
      {sh, 1, 5, 256, 255}, // Mem_Rb
      {sh, 2, 4, 256, 255}, // Mem_Displacement
      {sh, 2, 4, 512, 255}, // Mfc_Function
      {sh, 1, 5, 512, 255}, // Mfc_Ra
      {sh, 1, 5, 512, 255}, // Mfc_Rb
      {sh, 1, 5, 512, 255}, // Mbr_Ra
      {sh, 1, 5, 256, 255}, // Mbr_Rb
      {sh, 2, 4, 256, 255}, // Mbr_Displacement
      {sh, 1, 1, 256, 255}, // Opr_Bit
      {sh, 1, 7, 256, 255}, // Opr_Function
      {sh, 1, 5, 256, 255}, // Opr_Ra
      {sh, 1, 5, 256, 255}, // Opr_Rb
      {sh, 1, 3, 512, 255}, // Opr_Unused
      {sh, 1, 8, 256, 255}, // Opr_Literal
      {sh, 1, 5, 256, 255}, // Opr_Rc
      {sh, 4, 3, 512, 255}, // Pcd_Function
      {sh, 4, 3, 512, 255}  // Nop_Skip
    },
    maps1 {
      {sh, 11, 6, 256, 255}, // OpCode
      {sh,  5, 5, 256, 255}, // Bra_Ra
      {sh, 14, 7, 256, 255}, // Bra_Displacement
      {sh, 14, 3, 256, 255}, // F_P_Function
      {sh,  5, 5, 256, 255}, // F_P_Ra
      {sh, 10, 5, 256, 255}, // F_P_Rb
      {sh, 10, 5, 256, 255}, // F_P_Rc
      {sh,  5, 5, 256, 255}, // Mem_Ra
      {sh,  5, 5, 256, 255}, // Mem_Rb
      {sh, 16, 4, 256, 255}, // Mem_Displacement
      {sh,  1, 5, 256, 255}, // Mbr_Rb
      {sh,  3, 4, 256, 255}, // Mbr_Displacement
      {sh,  3, 1, 256, 255}, // Opr_Bit
      {sh,  4, 7, 256, 255}, // Opr_Function
      {sh,  5, 5, 256, 255}, // Opr_Ra
      {sh, 10, 5, 256, 255}, // Opr_Rb
      {sh, 10, 8, 256, 255}, // Opr_Literal
      {sh,  6, 5, 256, 255}  // Opr_Rc
    },
    maps2 {
      {sh, 16, 6, 256, 255}, // OpCode
      {sh, 10, 5, 256, 255}, // Bra_Ra
      {sh, 15, 7, 256, 255}, // Bra_Displacement
      {sh, 10, 5, 256, 255}, // Mem_Ra
      {sh, 10, 5, 256, 255}, // Mem_Rb
      {sh, 18, 4, 256, 255}, // Mem_Displacement
      {sh, 11, 1, 256, 255}, // Opr_Bit
      {sh, 16, 7, 256, 255}, // Opr_Function
      {sh,  5, 5, 256, 255}, // Opr_Ra
      {sh, 10, 5, 256, 255}, // Opr_Rb
      {sh, 14, 8, 256, 255}, // Opr_Literal
      {sh, 16, 5, 256, 255}  // Opr_Rc
    },
    maps3{
      {sh, 16, 6, 256, 255}, // OpCode
      {sh, 16, 5, 256, 255}, // Bra_Ra
      {sh, 15, 5, 256, 255}, // Mem_Ra
      {sh, 15, 5, 256, 255}, // Mem_Rb
      {sh, 18, 4, 256, 255}, // Mem_Displacement
      {sh, 16, 7, 256, 255}, // Opr_Function
      {sh, 10, 5, 256, 255}, // Opr_Ra
      {sh, 15, 5, 256, 255},  // Opr_Rb
      {sh, 11, 5, 256, 255}  // Opr_Rc
    },
    maps4{
      {sh, 16, 6, 256, 255}, // OpCode
      {sh, 16, 5, 256, 255}, // Bra_Ra
      {sh, 16, 5, 256, 255}, // Mem_Ra
      {sh, 16, 5, 256, 255}, // Mem_Rb
      {sh, 16, 5, 256, 255}, // Opr_Ra
      {sh, 16, 5, 256, 255}  // Opr_Rc
    },
    maps5{
      {sh, 17, 6, 256, 255}, // OpCode
      {sh, 17, 5, 256, 255}, // Mem_Ra
      {sh, 16, 5, 256, 255}  // Mem_Rb
    },
    maps6{
      {sh, 17, 6, 256, 255}, // OpCode
      {sh, 17, 5, 256, 255}, // Mem_Ra
      {sh, 17, 5, 256, 255}  // Mem_Rb
    },
    maps7{
      {sh, 18, 6, 256, 255}, // OpCode
      {sh, 18, 5, 256, 255}, // Mem_Ra
    },
    maps8{
      {sh, 18, 6, 256, 255}  // OpCode
    },
    maps9{
      {sh, 17, 6, 256, 255}  // OpCode
    },
    maps10{
      {sh, 17, 6, 256, 255}  // OpCode
    }
  {}
  void mix(Mixer& m) {
    INJECT_SHARED_blockPos
    INJECT_SHARED_bpos
    if ((blockPos == 0) && (bpos == 0)) {
      state = State::OpCode;
      for (uint32_t i = 0; i < nMaps - 1; i++) {
        if (((maps_mask[i] >> State::OpCode) & 1) != 0)
          maps[i][map_state(i, State::OpCode)].setDirect(0);
      }
      count = 0;
      cache.fill(op = { 0 });
    }
    else {
      count++;
      INJECT_SHARED_y
      switch (state) {
        case State::OpCode: {
          op.Opcode += op.Opcode + y;
          if (count == 6) {
            op.relOpcode = DECAlpha::rel_op[op.Opcode];
            switch (op.Format = DECAlpha::formats[op.Opcode]) {
              case DECAlpha::Bra: {
                state = State::Bra_Ra;
                maps1[map_state(0, State::Bra_Ra)].setDirect(cache(1).Rc);
                maps2[map_state(1, State::Bra_Ra)].setDirect((cache(1).Ra << 5) | cache(1).Rc);
                maps3[map_state(2, State::Bra_Ra)].setDirect((cache(1).Opcode << 10) | (cache(1).Ra << 5) | cache(1).Rc);
                maps4[map_state(3, State::Bra_Ra)].set(hash(cache(1).Ra, cache(2).Ra, cache(3).Ra, cache(4).Ra));
                break;
              }
              case DECAlpha::F_P: {
                state = State::F_P_Function;
                maps1[map_state(0, State::F_P_Function)].setDirect(op.Opcode);
                break;
              }
              case DECAlpha::Mem: {
                state = State::Mem_Ra;
                maps1[map_state(0, State::Mem_Ra)].setDirect(cache(1).Ra);
                maps2[map_state(1, State::Mem_Ra)].setDirect((cache(1).Ra << 5) | cache(2).Ra);
                maps3[map_state(2, State::Mem_Ra)].setDirect((cache(1).Ra << 10) | (cache(2).Ra << 5) | cache(3).Ra);
                maps4[map_state(3, State::Mem_Ra)].set(hash(cache(1).Ra, cache(2).Ra, cache(3).Ra, cache(4).Ra));
                maps5[map_state(4, State::Mem_Ra)].set(hash(op.Opcode, cache(1).Opcode, cache(1).Ra, cache(2).Opcode, cache(2).Ra));
                maps6[map_state(5, State::Mem_Ra)].set(hash(op.Opcode, cache(1).Ra, cache(2).Ra, cache(2).Rb, cache(3).Ra, cache(4).Ra));
                maps7[map_state(6, State::Mem_Ra)].set(hash(op.Opcode, cache(1).Ra, cache(1).Rb, cache(2).Ra, cache(3).Ra, cache(4).Ra, cache(5).Ra));
                break;
              }
              case DECAlpha::Mfc: {
                state = State::Mfc_Function;
                break;
              }
              case DECAlpha::Mbr: {
                state = State::Mbr_Ra; // usually R26, R27 or R31
                break;
              }
              case DECAlpha::Opr: {
                state = State::Opr_Bit;
                maps1[map_state(0, State::Opr_Bit)].setDirect(op.relOpcode);
                maps2[map_state(1, State::Opr_Bit)].set(hash(op.Opcode, cache(1).Opcode, cache(1).Function));
                break;
              }
              case DECAlpha::Pcd: {
                state = State::Pcd_Function;
                break;
              }
              case DECAlpha::Nop: {
                state = State::Nop_Skip;
                break;
              }
            }
            count = 0;
          }
          break;
        }
        case State::Bra_Ra: {
          op.Ra += op.Ra + y;
          if (count == 5) {
            state = State::Bra_Displacement;
            count = 0;
            maps1[map_state(0, State::Bra_Displacement)].setDirect(cache(1).Opcode);
            maps2[map_state(1, State::Bra_Displacement)].setDirect((cache(1).Opcode << 6) | cache(2).Opcode);
          }
          break;
        }
        case State::Bra_Displacement: {
          op.Displacement += op.Displacement + y;
          if ((count % 7) == 0) {
            if (count < 21) {
              maps0[state].setDirect(count / 7);
              maps1[map_state(0, State::Bra_Displacement)].set(hash(count, cache(1).Opcode, op.Displacement));
              maps2[map_state(1, State::Bra_Displacement)].set(hash(count, cache(1).Opcode, cache(2).Opcode, op.Displacement));
            }
            else {
              state = State::OpCode;
              count = 0;
            }
          }
          break;
        }
        case State::F_P_Function: {
          op.Function = op.Function + y;
          if (count == 11) {
            state = State::F_P_Ra;
            count = 0;
            maps1[map_state(0, State::F_P_Ra)].setDirect(cache(1).Rc);
          }
          else if ((count % 3) == 0) {
            maps0[state].setDirect(count / 3);
            maps1[map_state(0, State::F_P_Function)].set(hash(count, op.Opcode, op.Function));
          }
          break;
        }
        case State::F_P_Ra: {
          op.Ra += op.Ra + y;
          if (count == 5) {
            state = State::F_P_Rb;
            count = 0;
            maps1[map_state(0, State::F_P_Rb)].setDirect((op.Ra << 5) | cache(1).Rc);
          }
          break;
        }
        case State::F_P_Rb: {
          op.Rb += op.Rb + y;
          if (count == 5) {
            state = State::F_P_Rc;
            count = 0;
            maps1[map_state(0, State::F_P_Rc)].setDirect((op.Ra << 5) | op.Rb);
          }
          break;
        }
        case State::F_P_Rc: {
          op.Rc += op.Rc + y;
          if (count == 5) {
            lastRc = op.Rc;
            state = State::OpCode;
            count = 0;
          }
          break;
        }
        case State::Mem_Ra: {
          op.Ra += op.Ra + y;
          if (count == 5) {
            state = State::Mem_Rb;
            count = 0;
            maps1[map_state(0, State::Mem_Rb)].setDirect(op.relOpcode);
            maps2[map_state(1, State::Mem_Rb)].setDirect((op.relOpcode << 5) | cache(1).Rb);           
            maps3[map_state(2, State::Mem_Rb)].setDirect((op.relOpcode << 10) | (cache(1).Rb << 5) | cache(2).Rb);           
            maps4[map_state(3, State::Mem_Rb)].set(hash(op.Opcode, cache(1).Rb, cache(2).Rb, cache(3).Rb));
            maps5[map_state(4, State::Mem_Rb)].set(hash(op.Opcode, cache(1).Rb, cache(2).Ra, cache(3).Ra));
            maps6[map_state(5, State::Mem_Rb)].set(hash(op.Opcode, cache(1).Opcode, cache(1).Ra, cache(1).Rb));
          }
          break;
        }
        case State::Mem_Rb: {
          op.Rb += op.Rb + y;
          if (count == 5) {
            state = State::Mem_Displacement;
            count = 0;
            maps1[map_state(0, State::Mem_Displacement)].setDirect(op.Opcode);
            maps2[map_state(1, State::Mem_Displacement)].set(hash(op.Opcode, op.Rb));
            maps3[map_state(2, State::Mem_Displacement)].set(hash(op.Opcode, op.Rb, cache(1).Literal));
          }
          break;
        }
        case State::Mem_Displacement: {
          op.Displacement += op.Displacement + y;
          if ((count & 3) == 0) {
            if (count < 16) {
              maps0[state].setDirect(count / 4);
              maps1[map_state(0, State::Mem_Displacement)].set(hash(count, op.Opcode, op.Displacement));
              maps2[map_state(1, State::Mem_Displacement)].set(hash(count, op.Opcode, op.Rb, op.Displacement));
              maps3[map_state(2, State::Mem_Displacement)].set(hash(count, op.Opcode, op.Rb, cache(1).Literal, op.Displacement));
            }
            else {
              state = State::OpCode;
              count = 0;
            }
          }
          break;
        }
        case State::Mfc_Function: {
          op.Function += op.Function + y;
          if ((count & 3) == 0) {
            if (count < 16)
              maps0[state].setDirect(count / 4);
            else {
              state = State::Mfc_Ra; // usually R31
              count = 0;
            }            
          }
          break;
        }
        case State::Mfc_Ra: {
          op.Ra += op.Ra + y;
          if (count == 5) {
            state = State::Mfc_Rb; // usually R31
            count = 0;
          }
          break;
        }
        case State::Mfc_Rb: {
          op.Rb += op.Rb + y;
          if (count == 5) {
            state = State::OpCode;
            count = 0;
          }
          break;
        }
        case State::Mbr_Ra: {
          op.Ra += op.Ra + y;
          if (count == 5) {
            state = State::Mbr_Rb;
            count = 0;
            maps1[map_state(0, State::Mbr_Rb)].setDirect(op.relOpcode);
          }
          break;
        }
        case State::Mbr_Rb: {
          op.Rb += op.Rb + y;
          if (count == 5) {
            state = State::Mbr_Displacement;
            count = 0;
            maps1[map_state(0, State::Mbr_Displacement)].setDirect(op.relOpcode);
          }
          break;
        }
        case State::Mbr_Displacement: {
          op.Displacement += op.Displacement + y;
          if ((count & 3) == 0) {
            if (count < 16) {
              maps0[state].setDirect(count / 4);
              maps1[map_state(0, State::Mbr_Displacement)].setDirect(((count >> 1) & 0x6) | op.relOpcode);
            }
            else {
              state = State::OpCode;
              count = 0;
            }
          }
          break;
        }
        case State::Opr_Bit: {
          op.Bit = y;
          state = State::Opr_Function;
          count = 0;
          maps1[map_state(0, State::Opr_Function)].setDirect((op.relOpcode << 1) | op.Bit);
          maps2[map_state(1, State::Opr_Function)].set(hash(op.Opcode, op.Bit, cache(1).Opcode, cache(1).Function));
          maps3[map_state(2, State::Opr_Function)].set(hash(op.Opcode, op.Bit, cache(1).Opcode, cache(2).Opcode, cache(3).Opcode));
          break;
        }
        case State::Opr_Function: {
          op.Function += op.Function + y;
          if (count == 7) {
            state = State::Opr_Ra;
            count = 0;
            maps1[map_state(0, State::Opr_Ra)].setDirect(cache(1).Ra);
            maps2[map_state(1, State::Opr_Ra)].setDirect(cache(1).Rc);
            maps3[map_state(2, State::Opr_Ra)].setDirect((cache(1).Ra << 5) | cache(2).Ra);
            maps4[map_state(3, State::Opr_Ra)].set(hash(op.Bit, cache(1).Ra, cache(2).Ra, cache(3).Ra, cache(4).Ra));
          }
          break;
        }
        case State::Opr_Ra: {
          op.Ra += op.Ra + y;
          if (count == 5) {
            count = 0;
            if (op.Bit == 0) {
              state = State::Opr_Rb;
              maps1[map_state(0, State::Opr_Rb)].setDirect((cache(1).Ra << 5) | cache(1).Rb);
              maps2[map_state(1, State::Opr_Rb)].setDirect((cache(1).Rb << 5) | cache(2).Ra);
              maps3[map_state(2, State::Opr_Rb)].setDirect((op.Ra << 10) | (cache(1).Ra << 5) | cache(2).Ra);
            }
            else {
              state = State::Opr_Literal;
              maps1[map_state(0, State::Opr_Literal)].setDirect((op.relOpcode << 7) | op.Function);
              maps2[map_state(1, State::Opr_Literal)].set(hash(op.Opcode, op.Function, cache(1).Opcode, cache(1).Function));
            }
          }
          break;
        }
        case State::Opr_Rb: {
          op.Rb += op.Rb + y;
          if (count == 5) {
            state = State::Opr_Unused;
            count = 0;
          }
          break;
        }
        case State::Opr_Unused: {
          if (count == 3) {
            state = State::Opr_Rc;
            count = 0;
            maps1[map_state(0, State::Opr_Rc)].setDirect((op.Ra << 1) | op.Bit);
            maps2[map_state(1, State::Opr_Rc)].setDirect((op.Ra << 10) | (op.Rb << 5) | cache(1).Ra);
            maps3[map_state(2, State::Opr_Rc)].setDirect((cache(1).Ra << 5) | cache(2).Ra);
            maps4[map_state(3, State::Opr_Rc)].setDirect((cache(1).Ra << 10) | (cache(2).Ra << 5) | cache(3).Ra);
          }
          break;
        }
        case State::Opr_Literal: {
          op.Literal += op.Literal + y;
          if (count == 8) {
            state = State::Opr_Rc;
            count = 0;
            maps1[map_state(0, State::Opr_Rc)].setDirect((op.Ra << 1) | op.Bit);
            maps2[map_state(1, State::Opr_Rc)].setDirect(0x8000 | (op.Ra << 10) | (cache(1).Ra << 5) | cache(1).Rc);
            maps3[map_state(2, State::Opr_Rc)].setDirect(0x0400 | (cache(1).Ra << 5) | cache(2).Ra);
            maps4[map_state(3, State::Opr_Rc)].setDirect(0x8000 | (cache(1).Ra << 10) | (cache(2).Ra << 5) | cache(3).Ra);
          }
          break;
        }
        case State::Opr_Rc: {
          op.Rc += op.Rc + y;
          if (count == 5) {
            lastRc = op.Rc;
            state = State::OpCode;
            count = 0;
          }
          break;
        }
        case State::Pcd_Function:
        case State::Nop_Skip: {
          op.Function += op.Function + y;
          if (count == 26) {
            state = State::OpCode;
            count = 0;
          }
          else if ((count % 3) == 0)
            maps0[state].setDirect(count / 3);
          break;
        }
      }
    }
    if (count == 0) {
      maps0[state].setDirect(0);
      if (state == State::OpCode) {
        uint64_t ctx = hash(op.Opcode, op.Function);
        maps1[map_state(0, State::OpCode)].set(ctx);
        maps2[map_state(1, State::OpCode)].set(hash(ctx, cache(1).Opcode, cache(1).Function));
        maps3[map_state(2, State::OpCode)].set(ctx = hash(ctx, cache(1).Opcode, cache(2).Opcode));
        maps4[map_state(3, State::OpCode)].set(ctx = hash(ctx, cache(3).Opcode));
        maps5[map_state(4, State::OpCode)].set(hash(ctx, cache(4).Opcode));
        maps6[map_state(5, State::OpCode)].set(ctx = hash(op.Opcode, cache(1).Opcode, cache(2).Opcode, cache(3).Opcode, cache(4).Opcode, cache(5).Opcode));
        maps7[map_state(6, State::OpCode)].set(ctx = hash(ctx, cache(6).Opcode));
        maps8[map_state(7, State::OpCode)].set(ctx = hash(ctx, cache(7).Opcode));
        maps9[map_state(8, State::OpCode)].set(hash(op.Opcode, op.Ra, cache(1).Opcode, cache(1).Ra, cache(2).Opcode, cache(2).Ra));
        maps10[map_state(9, State::OpCode)].set(hash(op.Opcode, op.Rb, lastRc, cache(2).Opcode, cache(2).Rb, cache(3).Opcode, cache(3).Ra));
        last[op.Format] = op;
        cache.add(op);
        op = { 0 };
      }
    }    
   
    maps0[state].mix(m);
    for (uint32_t i = 0; i < nMaps - 1; i++) {
      if (((maps_mask[i] >> state) & 1) != 0)
        maps[i][map_state(i, state)].mix(m);
      else
        for (int j = 0; j < IndirectMap::MIXERINPUTS; j++)
          m.add(0);
    }

    uint8_t const opcode = (state != State::OpCode) ? op.Opcode : cache(1).Opcode;

    m.set(static_cast<uint32_t>(state) * 26 + count, State::Count * 26);
    m.set((state << 6) | opcode, State::Count * 64);
    m.set(finalize64(hash(state, count, opcode), 11), 2048);
    m.set(finalize64(hash(state, count, op.Opcode, cache(1).Opcode), 12), 4096);
    m.set(finalize64(hash(state, count, cache(1).Opcode, cache(2).Opcode), 12), 4096);
    m.set(finalize64(hash(state, count, cache(1).Opcode, cache(2).Opcode, cache(3).Opcode), 13), 8192);
    m.set(finalize64(hash(state, count, op.Opcode, cache(1).Opcode, cache(2).Opcode, cache(3).Opcode), 13), 8192);
    m.set(finalize64(hash(state, opcode, cache(1).Format, cache(2).Format, cache(3).Format, cache(4).Format), 13), 8192);
    m.set(finalize64(hash(state, count, op.Opcode, op.Bit, op.Ra), 12), 4096);
    m.set(finalize64(hash(state, op.Ra, op.Rb, op.Bit), 12), 4096);
    m.set(finalize64(hash(state, op.Ra, last[DECAlpha::Mem].Ra, cache(1).Format), 12), 4096);

    shared->State.DEC.state = state;
    shared->State.DEC.bcount = count;
  }
};
