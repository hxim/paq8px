#include "MTFList.hpp"

MTFList::MTFList(const uint16_t n) : root(0), Index(0), previous(n), next(n) {
  assert(n > 0);
  for( int i = 0; i < n; i++ ) {
    previous[i] = i - 1;
    next[i] = i + 1;
  }
  next[n - 1] = -1;
}

int MTFList::getFirst() {
  return Index = root;
}

int MTFList::getNext() {
  if( Index >= 0 ) {
    Index = next[Index];
  }
  return Index;
}

void MTFList::moveToFront(const int i) {
  assert(uint32_t(i) < previous.size());
  if((Index = i) == root ) {
    return;
  }
  const int p = previous[Index];
  const int n = next[Index];
  if( p >= 0 ) {
    next[p] = next[Index];
  }
  if( n >= 0 ) {
    previous[n] = previous[Index];
  }
  previous[root] = Index;
  next[Index] = root;
  root = Index;
  previous[root] = -1;
}