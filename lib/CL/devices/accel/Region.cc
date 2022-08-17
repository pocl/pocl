#include "Region.hh"

Region::~Region() {}

bool Region::isInRange(size_t dst) {
  return ((dst >= PhysAddress) && (dst < (PhysAddress + Size)));
}
