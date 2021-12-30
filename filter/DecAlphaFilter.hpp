#pragma once

#include "DecAlpha.hpp"
#include "../file/File.hpp"
#include "../Encoder.hpp"
#include "Filter.hpp"
#include <cstdint>

class DECAlphaFilter : public Filter {
private:
  constexpr static std::uint64_t block = 0x10000u; /**< block size */
public:
  /**
   * @todo Large file support
   * @param in
   * @param out
   * @param size
   * @param info
   */
  void encode(File* in, File* out, uint64_t size, int info, int& headerSize) override {
    Array<std::uint8_t> blk(block);
    for (std::uint64_t offset = 0u; offset < size; offset += block) {
      std::uint64_t length = std::min<std::uint64_t>(size - offset, block), bytesRead = static_cast<int>(in->blockRead(&blk[0u], length));
      if (bytesRead != length)
        quit("encodeDECAlpha read error");
      for (std::size_t i = 0u; i < static_cast<std::size_t>(bytesRead) - 3u; i += 4u) {
        std::uint32_t instruction = blk[i] | (blk[i + 1u] << 8u) | (blk[i + 2u] << 16u) | (blk[i + 3u] << 24u);
        if ((instruction >> 21u) == (0x34u << 5u) + 26u) { // bsr r26, offset
          std::uint32_t addr = instruction & 0x1FFFFFu;
          addr += static_cast<std::uint32_t>(offset + i) / 4u;
          instruction &= ~0x1FFFFFu;
          instruction |= addr & 0x1FFFFFu;
        }
        DECAlpha::Shuffle(instruction);
        blk[i] = instruction;
        blk[i + 1u] = instruction >> 8u;
        blk[i + 2u] = instruction >> 16u;
        blk[i + 3u] = instruction >> 24u;
      }
      out->blockWrite(&blk[0u], bytesRead);
    }
  }

  /**
   * @todo Large file support
   * @param in
   * @param out
   * @param fMode
   * @param size
   * @param diffFound
   * @return
   */
  uint64_t decode(File* in, File* out, FMode fMode, uint64_t size, uint64_t& diffFound) override {
    Array<std::uint8_t> blk(block);
    for (std::uint64_t offset = 0u; offset < size; offset += block) {
      std::uint64_t length = std::min<std::uint64_t>(size - offset, block);
      for (std::size_t i = 0u; i < static_cast<std::size_t>(length) - 3u; i += 4u) {
        blk[i]      = encoder->decompressByte(&encoder->predictorMain);
        blk[i + 1u] = encoder->decompressByte(&encoder->predictorMain);
        blk[i + 2u] = encoder->decompressByte(&encoder->predictorMain);
        blk[i + 3u] = encoder->decompressByte(&encoder->predictorMain);
        std::uint32_t instruction = (blk[i] | (blk[i + 1u] << 8u) | (blk[i + 2u] << 16u) | (blk[i + 3u] << 24u));
        DECAlpha::Unshuffle(instruction);
        if ((instruction >> 21u) == (0x34u << 5u) + 26u) { // bsr r26, offset
          std::uint32_t addr = instruction & 0x1FFFFFu;
          addr -= static_cast<std::uint32_t>(offset + i) / 4u;
          instruction &= ~0x1FFFFFu;
          instruction |= addr & 0x1FFFFFu;
        }
        blk[i] = instruction;
        blk[i + 1u] = instruction >> 8u;
        blk[i + 2u] = instruction >> 16u;
        blk[i + 3u] = instruction >> 24u;
      }
      std::size_t const l = static_cast<std::size_t>(length - (length & 3u));
      for (std::size_t i = 0u; i < static_cast<std::size_t>(length & 3u); i++)
        blk[l + i] = encoder->decompressByte(&encoder->predictorMain);

      if (fMode == FDECOMPRESS) {
        out->blockWrite(&blk[0u], length);
        encoder->printStatus();
      }
      else if (fMode == FCOMPARE) {
        for (std::size_t i = 0u; i < static_cast<std::size_t>(length); i++) {
          if (blk[i] != out->getchar()) {
            diffFound = offset + i;
            return 0u;
          }
        }
      }
    }
    return size;
  }
};
