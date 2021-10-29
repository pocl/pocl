/* Tests clCreateProgramWithBinary().

   Copyright (c) 2010-2021 pocl developers

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32
#define MAX_BINARIES  32
char kernel[] = "__kernel void k() {\n    return;\n}";

int
main(void){
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES + 1]; // + 1 for duplicate test
  cl_device_id device_id0;
  cl_uint num_devices;
  size_t i;
  size_t num_binaries;
  const unsigned char **binaries = NULL;
  size_t *binary_sizes = NULL;
  size_t num_bytes_copied;
  cl_int binary_statuses[MAX_BINARIES];
  cl_int binary_statuses2[MAX_BINARIES];
  cl_program program = NULL;
  cl_program program_with_binary = NULL;

  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");
  if (!nplatforms)
    return EXIT_FAILURE;
  
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                      devices, &num_devices);
  CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

  cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  size_t kernel_size = strlen(kernel);
  char* kernel_buffer = kernel;

  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer, 
				      &kernel_size, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram");
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 0, 0, &num_binaries);
  CHECK_OPENCL_ERROR_IN("clGetProgramInfo");

  num_binaries = num_binaries/sizeof(size_t);
  binary_sizes = (size_t*)malloc(num_binaries * sizeof(size_t));
  binaries = (const unsigned char**)calloc(num_binaries, sizeof(unsigned char*));

  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 
			 num_binaries*sizeof(size_t), binary_sizes , 
			 &num_bytes_copied);
  CHECK_OPENCL_ERROR_IN("clGetProgramInfo");
  
  for (i = 0; i < num_binaries; ++i) 
    binaries[i] = (const unsigned char*) malloc(binary_sizes[i] *
						sizeof(const unsigned char));

  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 
			 num_binaries*sizeof(char*), binaries, &num_bytes_copied);
  CHECK_OPENCL_ERROR_IN("clGetProgramInfo");
  
  cl_uint num = num_binaries < num_devices ? num_binaries : num_devices;
  if (num == 0)
    {
      err = !CL_SUCCESS;
      goto FREE_AND_EXIT;
    }
  
  program_with_binary = clCreateProgramWithBinary(context, num, devices, binary_sizes, 
						  binaries, binary_statuses, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithBinary");

  for (i = 0; i < num; ++i) {
      cl_program_binary_type bin_type = 0;
      err = clGetProgramBuildInfo(program_with_binary, devices[i],
                                  CL_PROGRAM_BINARY_TYPE,
                                  sizeof(bin_type), (void *)&bin_type,
                                  NULL);
      CHECK_OPENCL_ERROR_IN("get program binary type");

      /* cl_program_binary_type */
      switch(bin_type) {
        case CL_PROGRAM_BINARY_TYPE_NONE: /*0x0*/
          fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_NONE\n");
        break;
        case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT: /*0x1*/
          fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT\n");
        break;
        case CL_PROGRAM_BINARY_TYPE_LIBRARY: /*0x2*/
          fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_LIBRARY\n");
        break;
        case CL_PROGRAM_BINARY_TYPE_EXECUTABLE: /*0x4*/
          fprintf(stderr, "program binary type: CL_PROGRAM_BINARY_TYPE_EXECUTABLE\n");
         break;
      }
  }
  err = clReleaseProgram(program_with_binary);
  CHECK_OPENCL_ERROR_IN("clReleaseProgram");

  for (i = 0; i < num; i++)
    {
      if (binary_statuses[i] != CL_SUCCESS)
        {
          err = !CL_SUCCESS;
          goto FREE_AND_EXIT;
        }
    }
    
  // negative test1: invalid device
  device_id0 = devices[0];
  devices[0] = NULL; // invalid device
  program_with_binary = clCreateProgramWithBinary(context, num, devices, binary_sizes, 
						  binaries, binary_statuses, &err);

  if (err != CL_INVALID_DEVICE || program_with_binary != NULL)
    {
      err = !CL_SUCCESS;
      goto FREE_AND_EXIT;
    }
  err = CL_SUCCESS;

  devices[0] = device_id0;
  for (i = 0; i < num_binaries; ++i) free((void*)binaries[i]);
  free(binary_sizes);
  free(binaries);
  
  // negative test2: duplicate device
  num_binaries = 2;
  devices[1] = devices[0]; // duplicate
  
  binary_sizes = (size_t*)malloc(num_binaries * sizeof(size_t));
  binaries = (const unsigned char**)calloc(num_binaries, sizeof(unsigned char*));
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 1*sizeof(size_t), 
			 binary_sizes , &num_bytes_copied);
  CHECK_OPENCL_ERROR_IN("clGetProgramInfo");
  
  binary_sizes[1] = binary_sizes[0];
  
  binaries[0] = (const unsigned char*) malloc(binary_sizes[0] *
					      sizeof(const unsigned char));
  binaries[1] = (const unsigned char*) malloc(binary_sizes[1] *
					      sizeof(const unsigned char));
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 1 * sizeof(char*), 
			 binaries, &num_bytes_copied);
  CHECK_OPENCL_ERROR_IN("clGetProgramInfo");
  
  memcpy((void*)binaries[1], (void*)binaries[0], binary_sizes[0]);      
  program_with_binary = clCreateProgramWithBinary(context, 2, devices, binary_sizes, 
						  binaries, binary_statuses2, &err);
  if (err != CL_INVALID_DEVICE || program_with_binary != NULL)
    {
      err = !CL_SUCCESS;
      goto FREE_AND_EXIT;
    }
  err = CL_SUCCESS;

 FREE_AND_EXIT:  
  // Free resources
  for (i = 0; i < num_binaries; ++i) 
    if (binaries) 
      if(binaries[i]) 
	free((void*)binaries[i]);

  if (binary_sizes) 
    free(binary_sizes);
  if (binaries) 
    free(binaries);
  if (program)
    CHECK_CL_ERROR (clReleaseProgram (program));
  if (program_with_binary)
    CHECK_CL_ERROR (clReleaseProgram (program_with_binary));
  if (context)
    CHECK_CL_ERROR (clReleaseContext (context));

  CHECK_CL_ERROR (clUnloadCompiler ());

  return err == CL_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
