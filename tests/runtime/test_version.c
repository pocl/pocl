/* This test is part of pocl.
 * It is intended to run as the first test of the
 * testsuite, checking that the tests are 
 * not run against an other installed OpenCL library.
 */

#include "poclu.h"
#include <stdio.h>
#include <string.h>
#include "config.h"

int main(void)
{
	cl_context ctx;
	cl_device_id did;
	cl_platform_id pid; 
	cl_command_queue queue;
	cl_int rv;
	size_t rvs;
	char result[1024];
	char *needle;

	/* Check that the default platform we get from the ICD loader
	 * matches the pocl version string this binary was built against. */
	poclu_get_any_device( &ctx, &did, &queue);
	rv = clGetDeviceInfo( did, CL_DEVICE_PLATFORM, 
	                      sizeof(cl_platform_id), &pid, NULL);

	rv |= clGetPlatformInfo( pid, CL_PLATFORM_VERSION, 
	                        sizeof(result), result, &rvs);
	if( rv != CL_SUCCESS )
		return 1;
	result[rvs]=0;	// spec doesn't say it is null-terminated.
	if( strcmp( result, 
	            "OpenCL " POCL_CL_VERSION " pocl " PACKAGE_VERSION ", LLVM " LLVM_VERSION) != 0 ) {
		printf("Error: platform is: %s\n", result);
		return 2;
	}


	/* Pocl devices have the form 'type'-'details', if details are
	 * available. If not, they are of the form 'type'.
	 * print here only the type, as the details will be computer
	 * dependent */
	rv = clGetDeviceInfo( did, CL_DEVICE_NAME, 
	                      sizeof(result), result, NULL);
	if( rv != CL_SUCCESS )
		return 3;
	result[rvs]=0;
	needle = strchr(result, '-');
	if( needle != NULL ){
		*needle=0;		
	}
	printf("%s\n", result);
	

	return 0;
}

