#pragma once

#include <cstdint>

// Function eccCompute(), edcCompute() and eccedcInit() taken from
// ** UNECM - Decoder for ECM (Error code Modeler) format.
// ** version 1.0
// ** Copyright (c) 2002 Neill Corlett

/* LUTs used for computing ECC/EDC */
static uint8_t eccFLut[256];
static uint8_t eccBLut[256];
static uint32_t edcLut[256];
static bool tablesInit = false;

static void eccedcInit() {
  if( tablesInit ) {
    return;
  }
  uint32_t i = 0;
  uint32_t j = 0;
  uint32_t edc = 0;
  for( i = 0; i < 256; i++ ) {
    j = (i << 1U) ^ ((i & 0x80U) != 0u ? 0x11DU : 0U);
    eccFLut[i] = j;
    eccBLut[i ^ j] = i;
    edc = i;
    for( j = 0; j < 8; j++ ) {
      edc = (edc >> 1U) ^ ((edc & 1U) != 0u ? 0xD8018001 : 0);
    }
    edcLut[i] = edc;
  }
  tablesInit = true;
}

static void eccCompute(const uint8_t *src, uint32_t majorCount, uint32_t minorCount, uint32_t majorMult, uint32_t minorInc, uint8_t *dest) {
  uint32_t size = majorCount * minorCount;
  uint32_t major = 0;
  uint32_t minor = 0;
  for( major = 0; major < majorCount; major++ ) {
    uint32_t index = (major >> 1U) * majorMult + (major & 1U);
    uint8_t eccA = 0;
    uint8_t eccB = 0;
    for( minor = 0; minor < minorCount; minor++ ) {
      uint8_t temp = src[index];
      index += minorInc;
      if( index >= size ) {
        index -= size;
      }
      eccA ^= temp;
      eccB ^= temp;
      eccA = eccFLut[eccA];
    }
    eccA = eccBLut[eccFLut[eccA] ^ eccB];
    dest[major] = eccA;
    dest[major + majorCount] = eccA ^ eccB;
  }
}

static auto edcCompute(const uint8_t *src, int size) -> uint32_t {
  uint32_t edc = 0;
  while((size--) != 0 ) {
    edc = (edc >> 8U) ^ edcLut[(edc ^ (*src++)) & 0xFFU];
  }
  return edc;
}
