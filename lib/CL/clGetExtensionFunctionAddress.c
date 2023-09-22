#include "pocl_cl.h"

#include <string.h>

/* Note - this is deprecated in 1.1, but (some of) the ICD loaders are built
 * against OCL 1.1, so we need it.
 */ 
POCL_EXPORT CL_API_ENTRY void * CL_API_CALL
POname(clGetExtensionFunctionAddress)(const char * func_name ) 
CL_API_SUFFIX__VERSION_1_0
{

  cl_platform_id pocl_platform;
  cl_uint actual_num = 0;
  POname (clGetPlatformIDs) (1, &pocl_platform, &actual_num);
  if (actual_num != 1)
    {
      POCL_MSG_WARN ("Couldn't get the platform ID of PoCL platform\n");
      return NULL;
    }

  return POname (
      clGetExtensionFunctionAddressForPlatform (pocl_platform, func_name));
}
POsymAlways(clGetExtensionFunctionAddress)
