#include <assert.h>
#include <string.h>
#include "pocl_cl.h"

/*
 * Provide the ICD loader the specified function to get the pocl platform.
 * 
 * TODO: the functionality of this seems the same as that of clGetPlatformIDs.
 * but we cannot call that, as it is defined in the ICD loader itself.
 * 
 */ 
extern struct _cl_platform_id _platforms[1];

CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint           num_entries,
                       cl_platform_id *  platforms,
                       cl_uint *         num_platforms) CL_API_SUFFIX__VERSION_1_0
{	
  int const num = 1;
  int i;
  
  if (platforms != NULL) {
    if (num_entries < num)
      return CL_INVALID_VALUE;
    
    for (i=0; i<num; ++i)
      platforms[i] = &_platforms[i];
  }
  
  if (num_platforms != NULL)
    *num_platforms = num;
  
  return CL_SUCCESS;
}
