/* OpenCL runtime library: clGetPlatformIDs()
 * 
 * Copyright - Kalle Raiskila 2011.
 * 
 * This is file is licencsed under a "Free Beer" type license:
 * You can do whatever you want with this stuff. If we meet some day, 
 * and you think this stuff is worth it, you can buy me a beer in return.
 */


#include <assert.h>
#include <string.h>
#include "pocl_cl.h"

static struct _cl_platform_id _platforms[1]  = {{42}};

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint           num_entries,
                 cl_platform_id *  platforms,
                 cl_uint *         num_platforms) CL_API_SUFFIX__VERSION_1_0
{	

	// user requests only number of platforms - not types
	if (num_entries == 0 && num_platforms != NULL)
      {
		*num_platforms = 1;
		return CL_SUCCESS;
      }

	// Bad request - no place to store response
	if (num_entries > 0 && platforms == NULL)
      return CL_INVALID_VALUE;
	
	// Check required by spec
	if (platforms == NULL && num_platforms == NULL)
      return CL_INVALID_VALUE;
	
	// platform is not used now - just mark this platform as 'valid'
	platforms[0] = &(_platforms[0]);
	
	if (num_platforms != NULL)
      *num_platforms = 1;

	return CL_SUCCESS;
}
