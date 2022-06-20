#include <assert.h>
#include <string.h>
#include "pocl_cl.h"

/*
 * GetPlatformIDs that support ICD.
 * This function is required by the ICD specification.
 */ 

#ifdef BUILD_ICD
POCL_EXPORT CL_API_ENTRY cl_int CL_API_CALL
POname(clIcdGetPlatformIDsKHR)(cl_uint           num_entries,
                       cl_platform_id *  platforms,
                       cl_uint *         num_platforms) CL_API_SUFFIX__VERSION_1_0
{	
  return POname(clGetPlatformIDs)( num_entries, platforms, num_platforms );
}
POsymICD(clIcdGetPlatformIDsKHR)
#endif

