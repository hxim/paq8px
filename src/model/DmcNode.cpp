#include "DmcNode.hpp"
#include <cassert>


uint8_t DMCNode::getState() const { return uint8_t(((_nx0 & 0x0f) << 4) | (_nx1 & 0x0f)); }

void DMCNode::setState(const uint8_t state) {
  _nx0 = (_nx0 & 0xfffffff0) | (state >> 4);
  _nx1 = (_nx1 & 0xfffffff0) | (state & 0x0f);
}

uint32_t  DMCNode::getNx0() const { return _nx0 >> 4; }

void DMCNode::setNx0(const uint32_t nx0) {
  assert((nx0 >> 28) == 0);
  _nx0 = (_nx0 & 0x0f) | (nx0 << 4);
}

uint32_t  DMCNode::getNx1() const { return _nx1 >> 4; }

void DMCNode::setNx1(const uint32_t nx1) {
  assert((nx1 >> 28) == 0);
  _nx1 = (_nx1 & 0x0f) | (nx1 << 4);
}
