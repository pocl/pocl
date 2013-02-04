#include <fenv.h>

void cl_set_rounding_mode(int mode)
{
  fesetround(mode);
}

int cl_get_default_rounding_mode()
{
  return fegetround();
}
