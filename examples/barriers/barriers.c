/* barriers - OpenCL synchronization barriers example.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

#define N 4

int
main (void)
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
  size_t global_work_size[1];
  size_t local_work_size[1];

  source_file = fopen (SRCDIR "/barriers.cl", "r");
  fseek (source_file, 0, SEEK_END);
  source_size = ftell (source_file);
  fseek (source_file, 0, SEEK_SET);

  source = (char *) malloc (source_size);
  assert (source != NULL);

  fread (source, source_size, 1, source_file);

  fclose(source_file);
  
  context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, 
				    NULL, NULL, NULL); 
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
 
  kernel = clCreateKernel(program, "barriers", NULL); 
  if (kernel == (cl_kernel)0) 
    { 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 

  global_work_size[0] = N; 
  local_work_size[0]= 2; 
 
  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, 
			       global_work_size, local_work_size,  
			       0, NULL, NULL); 

  clReleaseKernel(kernel); 
  clReleaseProgram(program); 
  clReleaseCommandQueue(cmd_queue); 
  clReleaseContext(context); 

  return 0;
}
