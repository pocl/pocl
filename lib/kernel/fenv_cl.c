#include <fenv.h>

void cl_set_rounding_mode(int mode)
{
  fesetround(mode);
}

void cl_set_default_rounding_mode()
{
  fesetenv(FE_DFL_ENV);
}
