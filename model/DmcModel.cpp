#include "DmcModel.hpp"

auto DmcModel::incrementCounter(const uint32_t x, const uint32_t increment) -> uint32_t {
  return (((x << 6) - x) >> 6) + (increment << 10); // x * (1-1/64) + increment
}

DmcModel::DmcModel(const Shared* const sh, const uint64_t dmcNodes, const uint32_t thStart) : shared(sh), 
  t(min(dmcNodes + dmcNodesBase, dmcNodesMax)),
  sm(sh, 1, 256, 256 /*64-512 are all fine*/, StateMap::BitHistory) {
    resetStateGraph(thStart);
  }

void DmcModel::resetStateGraph(const uint32_t thStart) {
  // the top 4 bits must be unused by nx0 and nx1 for storing the 4+4 bits of the bit history state byte
  assert(((t.size() - 1) >> 28) == 0);
  top = curr = extra = 0;
  threshold = thStart;
  thresholdFine = thStart << 11;
  for( int j = 0; j < 256; ++j ) { //256 trees
    for( int i = 0; i < 255; ++i ) { //255 nodes in each tree
      if( i < 127 ) { //internal tree nodes
        t[top].setNx0(top + i + 1); // left node
        t[top].setNx1(top + i + 2); // right node
      } else { // 128 leaf nodes - they each references a root node of tree(i)
        int linkedTreeRoot = (i - 127) * 2 * 255;
        t[top].setNx0(linkedTreeRoot); // left node  -> root of tree 0,2,4...
        t[top].setNx1(linkedTreeRoot + 255); // right node -> root of tree 1,3,5...
      }
      t[top].c0 = t[top].c1 = thStart < 1024 ? 2048 : 512; // 2.0  0.5
      t[top].setState(0);
      top++;
    }
  }
}

void DmcModel::update() {
  uint32_t c0 = t[curr].c0;
  uint32_t c1 = t[curr].c1;
  INJECT_SHARED_y
  const uint32_t n = y == 0 ? c0 : c1;

  // update counts, state
  t[curr].c0 = incrementCounter(c0, 1 - y);
  t[curr].c1 = incrementCounter(c1, y);
  t[curr].setState(StateTable::next(t[curr].getState(), y, rnd));

  // clone next state when threshold is reached
  if( n > threshold ) {
    const uint32_t next = y == 0 ? t[curr].getNx0() : t[curr].getNx1();
    c0 = t[next].c0;
    c1 = t[next].c1;
    const uint32_t nn = c0 + c1;

    if( nn > n + threshold ) {
      if( top != t.size()) { // state graph is not yet full, let's clone
        uint32_t c0Top = uint64_t(c0) * uint64_t(n) / uint64_t(nn);
        uint32_t c1Top = uint64_t(c1) * uint64_t(n) / uint64_t(nn);
        assert(c0 >= c0Top);
        assert(c1 >= c1Top);
        c0 -= c0Top;
        c1 -= c1Top;

        t[top].c0 = c0Top;
        t[top].c1 = c1Top;
        t[next].c0 = c0;
        t[next].c1 = c1;

        t[top].setNx0(t[next].getNx0());
        t[top].setNx1(t[next].getNx1());
        t[top].setState(t[next].getState());
        if( y == 0 ) {
          t[curr].setNx0(top);
        } else {
          t[curr].setNx1(top);
        }

        ++top;

        if( threshold < 8 * 1024 ) {
          threshold = (++thresholdFine) >> 11;
        }
      } else { // state graph was full
        extra += nn >> 10;
      }
    }
  }

  if( y == 0 ) {
    curr = t[curr].getNx0();
  } else {
    curr = t[curr].getNx1();
  }
}

auto DmcModel::isFull() const -> bool {
  return (extra >> 7) > uint32_t(t.size());
}

auto DmcModel::st1() const -> int {
  const uint32_t n0 = t[curr].c0 + 1;
  const uint32_t n1 = t[curr].c1 + 1;
  return stretch((n1 << 12) / (n0 + n1));
}

auto DmcModel::st2() -> int {
  const uint8_t state = t[curr].getState();
  if (state == 0)
    return 0; // p = 0.5
  else
    return stretch(sm.p1(state));
}

auto DmcModel::stw() -> int {
  shared->GetUpdateBroadcaster()->subscribe(this);
  return st1() * 5 + st2() * 3; // (weighted) average of the predictions for stability
}
