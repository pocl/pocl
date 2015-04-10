#include "test_include.h"

__kernel void testk()
{
  test_include();
  printf(PRINT_DEFINE);
}
