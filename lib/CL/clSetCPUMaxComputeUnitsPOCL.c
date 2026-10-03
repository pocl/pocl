/* OpenCL runtime library: clSetCPUMaxComputeUnitsPOCL

   Copyright (c) 2026 Tim Besard

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "devices/devices.h"
#include "pocl_cl.h"

#include <limits.h>

/* Lets an application that embeds PoCL choose the size of the CPU devices'
 * thread pools without changing the process environment, which isn't
 * thread-safe and is inherited by child processes. The count applies to
 * each CPU device, like POCL_CPU_MAX_CU_COUNT. It can be set repeatedly
 * until the devices start initializing, after which it's an error. */
CL_API_ENTRY cl_int CL_API_CALL
POname (clSetCPUMaxComputeUnitsPOCL) (cl_platform_id platform,
                                      cl_uint num_compute_units)
CL_API_SUFFIX__VERSION_1_2
{
  cl_platform_id pocl_platform;
  POname (clGetPlatformIDs) (1, &pocl_platform, NULL);
  POCL_RETURN_ERROR_ON ((platform != pocl_platform), CL_INVALID_PLATFORM,
                        "Can only configure the POCL platform\n");

  /* the drivers store the count in an int */
  POCL_RETURN_ERROR_ON ((num_compute_units == 0 || num_compute_units > INT_MAX),
                        CL_INVALID_VALUE,
                        "Invalid number of compute units: %u\n",
                        num_compute_units);

  cl_int errcode = pocl_set_cpu_max_compute_units (num_compute_units);
  POCL_RETURN_ERROR_ON ((errcode != CL_SUCCESS), errcode,
                        "The devices have already been initialized\n");

  return CL_SUCCESS;
}
POsym (clSetCPUMaxComputeUnitsPOCL)
