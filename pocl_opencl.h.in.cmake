#cmakedefine RENAME_POCL

#ifdef RENAME_POCL
#include "rename_opencl.h"
#endif

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/opencl.h>
#endif

#include <poclu.h>
