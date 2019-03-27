#cmakedefine ENABLE_PROXY_DEVICE

#ifdef ENABLE_PROXY_DEVICE
#include "rename_opencl.h"
#endif

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/opencl.h>
#endif

#include <poclu.h>
