#include "Region.h"

Region::~Region(){}

bool
Region::isInRange(size_t dst) {
  return ((dst >= PhysAddress) && (dst < (PhysAddress + Size)));
}
