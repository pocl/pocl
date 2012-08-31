#include "pocl_cl.h"

#include <string.h>

/* Note - this is deprecated in 1.1, but (some of) the ICD loaders are built
 * against OCL 1.1, so we need it.
 */ 
CL_API_ENTRY void * CL_API_CALL 
POclGetExtensionFunctionAddress(const char * func_name ) 
CL_EXT_SUFFIX__VERSION_1_0
{

#ifdef BUILD_ICD
  if( strcmp(func_name, "clIcdGetPlatformIDsKHR")==0 )
    return (void *)&POclIcdGetPlatformIDsKHR;
#endif
  if( strcmp(func_name, "clGetPlatformInfo")==0 )
    return (void *)&POclGetPlatformInfo;
  
  return NULL;
}
POsymAlways(clGetExtensionFunctionAddress)
