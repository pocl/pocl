/* This test is part of pocl.
 * It is intended to run as the first test of the
 * testsuite, checking that the tests are 
 * not run against an other installed OpenCL library.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "poclu.h"
#include "config.h"
#include "pocl_version.h"

#define S(A) #A
#define STRINGIFY(X) S(X)

int main(void)
{
	cl_context context;
	cl_device_id did;
	cl_platform_id pid; 
	cl_command_queue queue;
	size_t rvs;
	char result[1024];
	char *needle;

	/* Check that the default platform we get from the ICD loader
	 * matches the pocl version string this binary was built against. */
	CHECK_CL_ERROR (poclu_get_any_device2 (&context, &did, &queue, &pid));
        TEST_ASSERT (context);
        TEST_ASSERT (did);
        TEST_ASSERT (queue);

	CHECK_CL_ERROR(clGetPlatformInfo( pid, CL_PLATFORM_VERSION,
				sizeof(result), result, &rvs));

	result[rvs]=0;	// spec doesn't say it is null-terminated.

        const char *expected = "OpenCL " STRINGIFY(POCL_PLATFORM_VERSION_MAJOR)
         "." STRINGIFY(POCL_PLATFORM_VERSION_MINOR) " PoCL " POCL_VERSION_FULL;
        if (strncmp (result, expected, strlen (expected)) != 0)
          {
            printf ("Error: platform is: %s\n", result);
            printf ("Should be: %s\n", expected);
            return 2;
          }


	/* Pocl devices have the form 'type'-'details', if details are
	 * available. If not, they are of the form 'type'.
	 * print here only the type, as the details will be computer
	 * dependent */
	CHECK_CL_ERROR(clGetDeviceInfo( did, CL_DEVICE_NAME,
			      sizeof(result), result, NULL));

	result[rvs]=0;
	needle = strchr(result, '-');
	if( needle != NULL ){
            *needle = 0;
        }
	printf("%s\n", result);

        CHECK_CL_ERROR (clReleaseCommandQueue (queue));
        CHECK_CL_ERROR (clReleaseContext (context));
        CHECK_CL_ERROR (clUnloadPlatformCompiler (pid));
        return 0;
}
