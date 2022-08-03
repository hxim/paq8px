#include "AudioModel.hpp"

AudioModel::AudioModel(Shared* const sh) : shared(sh) {}

int AudioModel::s2(const uint32_t i) const {
  INJECT_SHARED_buf
  return int(short(buf(i) + 256 * buf(i - 1)));
}

int AudioModel::t2(const uint32_t i) const {
  INJECT_SHARED_buf
  return int(short(buf(i - 1) + 256 * buf(i)));
}

int AudioModel::x1(const uint32_t i) const {
  INJECT_SHARED_buf
  switch( wMode ) {
    case 0:
      return buf(i) - 128;
    case 1:
      return buf(i << 1) - 128;
    case 2:
      return s2(i << 1);
    case 3:
      return s2(i << 2);
    case 4:
      return (buf(i) ^ 128) - 128;
    case 5:
      return (buf(i << 1) ^ 128) - 128;
    case 6:
      return t2(i << 1);
    case 7:
      return t2(i << 2);
    default:
      return 0;
  }
}

int AudioModel::x2(const uint32_t i) const {
  INJECT_SHARED_buf
  switch( wMode ) {
    case 0:
      return buf(i + s) - 128;
    case 1:
      return buf((i << 1) - 1) - 128;
    case 2:
      return s2((i + s) << 1);
    case 3:
      return s2((i << 2) - 2);
    case 4:
      return (buf(i + s) ^ 128) - 128;
    case 5:
      return (buf((i << 1) - 1) ^ 128) - 128;
    case 6:
      return t2((i + s) << 1);
    case 7:
      return t2((i << 2) - 2);
    default:
      return 0;
  }
}

int AudioModel::signedClip8(const int i) {
  return max(-128, min(127, i));
}

int AudioModel::signedClip16(const int i) {
  return max(-32768, min(32767, i));
}
