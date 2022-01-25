#include "Region.h"

bool
Region::isInRange(size_t dst) {
  return ((dst >= PhysAddress) && (dst < (PhysAddress + Size)));
}
