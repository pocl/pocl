/* regression test for GitHub issue #1747
   (Miscompilation with SPIR-V kernel)

   Copyright (c) 2024,2025 Tim Besard

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
  cl_long *ptr;
  size_t maxsize;
  struct
  {
    size_t dim1;
  } dims;
  size_t len;
} DeviceArray;

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      exit (1);                                                               \
    }

void
list_platforms ()
{
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs (0, NULL, &num_platforms);
  CHECK_ERROR (err);

  cl_platform_id *platforms
    = (cl_platform_id *)malloc (sizeof (cl_platform_id) * num_platforms);
  err = clGetPlatformIDs (num_platforms, platforms, NULL);
  CHECK_ERROR (err);

  printf ("Available platforms:\n");
  for (cl_uint i = 0; i < num_platforms; i++)
    {
      char platform_name[128];
      err = clGetPlatformInfo (platforms[i], CL_PLATFORM_NAME,
                               sizeof (platform_name), platform_name, NULL);
      CHECK_ERROR (err);
      printf ("%d: %s\n", i, platform_name);
    }

  free (platforms);
}

cl_platform_id
get_platform (int index)
{
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs (0, NULL, &num_platforms);
  CHECK_ERROR (err);

  if (index >= num_platforms)
    {
      printf ("Platform index %d is out of range (max %d) at %s:%d\n", index,
              num_platforms - 1, __FILE__, __LINE__);
      exit (1);
    }

  cl_platform_id *platforms
    = (cl_platform_id *)malloc (sizeof (cl_platform_id) * num_platforms);
  err = clGetPlatformIDs (num_platforms, platforms, NULL);
  CHECK_ERROR (err);

  cl_platform_id platform = platforms[index];
  free (platforms);
  return platform;
}

int
main (int argc, char **argv)
{
  list_platforms ();

  // Use first platform by default or take from command line
  int platform_index = 0;
  if (argc > 1)
    {
      platform_index = atoi (argv[1]);
    }

  const char *input_spirv = "test_issue_1747.spv";
  if (argc > 2)
    {
      input_spirv = argv[2];
    }

  cl_int err;
  cl_platform_id platform = get_platform (platform_index);
  cl_device_id device;

  // Get platform name
  char platform_name[128];
  err = clGetPlatformInfo (platform, CL_PLATFORM_NAME, sizeof (platform_name),
                           platform_name, NULL);
  CHECK_ERROR (err);
  printf ("Using platform %d: %s\n", platform_index, platform_name);

  // Get device
  err = clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  CHECK_ERROR (err);

  // Get device name
  char device_name[128];
  err = clGetDeviceInfo (device, CL_DEVICE_NAME, sizeof (device_name),
                         device_name, NULL);
  CHECK_ERROR (err);
  printf ("Device: %s\n", device_name);

  if (!poclu_device_supports_il (device, "SPIR-V_1.4"))
    {
      printf ("SKIP: The test requires support for SPIR-V 1.4\n");
      exit (77);
    }

  // Rest of the code remains the same...
  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR (err);

  cl_queue_properties props[] = { 0 };
  cl_command_queue queue
    = clCreateCommandQueueWithProperties (context, device, props, &err);
  CHECK_ERROR (err);

  // Create DeviceArray structure
  DeviceArray A;
  A.maxsize = 16;
  A.dims.dim1 = 2;
  A.len = 2;

  // Allocate SVM memory for the data array
  A.ptr = (cl_long *)clSVMAlloc (context, CL_MEM_READ_WRITE,
                                 sizeof (cl_long) * A.len, 0);
  if (A.ptr == NULL)
    {
      printf ("SVM allocation failed at %s:%d\n", __FILE__, __LINE__);
      return 1;
    }

  // Load SPIR-V binary
  FILE *f = fopen (input_spirv, "rb");
  if (!f)
    {
      printf ("Failed to open %s at %s:%d\n", input_spirv, __FILE__, __LINE__);
      return 1;
    }
  fseek (f, 0, SEEK_END);
  size_t size = ftell (f);
  fseek (f, 0, SEEK_SET);

  unsigned char *binary = (unsigned char *)malloc (size);
  fread (binary, 1, size, f);
  fclose (f);

  cl_program program = clCreateProgramWithIL (context, binary, size, &err);
  CHECK_ERROR (err);

  err = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t log_size;
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                             &log_size);
      char *log = (char *)malloc (log_size);
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, log_size,
                             log, NULL);
      printf ("Build error at %s:%d:\n%s\n", __FILE__, __LINE__, log);
      free (log);
      return 1;
    }

  cl_kernel kernel = clCreateKernel (
    program, "_Z6kernel13CLDeviceArrayI5Int64Li1ELi1EE", &err);
  CHECK_ERROR (err);

  // Inform runtime about SVM pointers
  void *svm_ptrs[] = { A.ptr };
  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                             sizeof (void *), svm_ptrs);
  CHECK_ERROR (err);

  // Set kernel argument
  err = clSetKernelArg (kernel, 0, sizeof (DeviceArray), &A);
  CHECK_ERROR (err);

  // Execute kernel
  size_t global_size = 2;
  size_t local_size = 2;

  printf ("Launching kernel\n");
  err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global_size,
                                &local_size, 0, NULL, NULL);
  CHECK_ERROR (err);

  printf ("Waiting for completion\n");
  err = clFinish (queue);
  CHECK_ERROR (err);

  // Cleanup
  clSVMFree (context, A.ptr);
  clReleaseKernel (kernel);
  clReleaseProgram (program);
  clReleaseCommandQueue (queue);
  clReleaseContext (context);
  free (binary);

  return 0;
}
