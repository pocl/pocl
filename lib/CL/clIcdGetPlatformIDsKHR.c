#include <assert.h>
#include <string.h>
#include "pocl_cl.h"

/*
 * GetPlatformIDs that support ICD.
 * This function is required by the ICD specification.
 */ 
extern struct _cl_platform_id _platforms[1];

#ifdef BUILD_ICD
CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint           num_entries,
                       cl_platform_id *  platforms,
                       cl_uint *         num_platforms) CL_API_SUFFIX__VERSION_1_0
{	
  return clGetPlatformIDs( num_entries, platforms, num_platforms );
}
#endif

