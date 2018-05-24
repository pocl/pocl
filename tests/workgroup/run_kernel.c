/* run_kernel - a generic launcher for a kernel without inputs and outputs

   Copyright (c) 2012 Pekka Jääskeläinen / TUT
   
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
#include <CL/opencl.h>
#include "poclu.h"

#ifdef _MSC_VER
#  include "vccompat.hpp"
#endif

/**
 * The test kernels are assumed to:
 *
 * 1) called 'test_kernel'
 * 2) no inputs or outputs, only work item id printfs to verify the correct 
 *    workgroup transformations
 * 3) executable with any local and global dimensions and sizes
 *
 * Usage:
 *
 * ./run_kernel somekernel.cl 2 2 3 4
 *
 * Where the first integer is the number of work groups to execute and the
 * rest are the local dimensions.
 */
int
main (int argc, char **argv)
{
  FILE *source_file;
  char *source;
  int source_size;
  cl_context context;
  size_t cb;
  cl_device_id *devices;
  cl_command_queue cmd_queue;
  cl_program program;
  cl_int err;
  cl_kernel kernel;
  size_t global_work_size[3];
  size_t local_work_size[3];
  char kernel_path[2048];

  snprintf (kernel_path, 2048,  "%s/%s", SRCDIR, argv[1]);
  source_file = fopen(kernel_path, "r");
  TEST_ASSERT (source_file != NULL && "Kernel .cl not found.");

  fseek (source_file, 0, SEEK_END);
  source_size = ftell (source_file);
  fseek (source_file, 0, SEEK_SET);

  source = (char *) malloc (source_size + 1);
  TEST_ASSERT (source != NULL);

  fread (source, source_size, 1, source_file);
  source[source_size] = '\0';

  fclose(source_file);

  local_work_size[0] = atoi(argv[3]);
  local_work_size[1] = atoi(argv[4]);
  local_work_size[2] = atoi(argv[5]);

  global_work_size[0] = local_work_size[0] * atoi(argv[2]);
  global_work_size[1] = local_work_size[1];
  global_work_size[2] = local_work_size[2];
  
  context = poclu_create_any_context();
  if (context == (cl_context)0) 
    return -1; 

  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb); 
  devices = (cl_device_id *) malloc(cb);
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL); 
 
  cmd_queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (cmd_queue == (cl_command_queue)0) 
    { 
      clReleaseContext(context); 
      free(devices); 
      return -1; 
    } 
  free(devices); 

  program = clCreateProgramWithSource(context, 
				      1, (const char**)&source, NULL, NULL); 
  if (program == (cl_program)0) 
    { 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  kernel = clCreateKernel(program, "test_kernel", NULL); 
  if (kernel == (cl_kernel)0) 
    { 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 


  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 3, NULL, 
			       global_work_size, local_work_size,  
			       0, NULL, NULL); 
  if(err != CL_SUCCESS)
    {
       clReleaseKernel(kernel);
       clReleaseProgram(program);
       clReleaseCommandQueue(cmd_queue);
       clReleaseContext(context);
       return -1;
    }
  clFinish(cmd_queue);
  clReleaseKernel(kernel); 
  clReleaseProgram(program); 
  clReleaseCommandQueue(cmd_queue); 
  clReleaseContext(context); 

  return 0;
}
