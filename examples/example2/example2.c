/* example2 - Matrix transpose example from OpenCL specification.

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
#include "poclu.h"

#ifdef _MSC_VER
#  include "vccompat.hpp"
#endif

#define WIDTH 256
#define HEIGHT 4096
#define PADDING 32

static void delete_memobjs(cl_mem *memobjs, int n) ;

int
main (void)
{
  FILE *source_file;
  char *source;
  int source_size;
  cl_float *input, *output;
  int i;
  int j;
  cl_context  context; 
  size_t cb;
  cl_device_id *devices;
  cl_command_queue cmd_queue;
  cl_program program;
  cl_int err;
  cl_kernel kernel;
  cl_mem memobjs[2];
  size_t global_work_size[2];
  size_t local_work_size[2];

  source_file = fopen("example2.cl", "r");
  if (source_file == NULL) 
    source_file = fopen (SRCDIR "/example2.cl", "r");

  assert(source_file != NULL && "example2.cl not found!");

  fseek (source_file, 0, SEEK_END);
  source_size = ftell (source_file);
  fseek (source_file, 0, SEEK_SET);

  source = (char *) malloc (source_size + 1);
  assert (source != NULL);

  fread (source, source_size, 1, source_file);
  source[source_size] = '\0';

  fclose (source_file);

  input = (cl_float *) malloc (WIDTH * HEIGHT * sizeof (cl_float));
  output = (cl_float *) malloc (WIDTH * (HEIGHT + PADDING) * sizeof (cl_float));

  srand48(0);
  for (i = 0; i < HEIGHT; ++i)
    {
      for (j = 0; j < WIDTH; ++j)
      input[i * WIDTH + j] = (cl_float)drand48();
    }
  
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

  memobjs[0] = clCreateBuffer(context,
			      CL_MEM_READ_WRITE,
			      sizeof(cl_float) * WIDTH * (HEIGHT + PADDING), NULL, NULL);
  if (memobjs[0] == (cl_mem)0) 
    { 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 

  memobjs[1] = clCreateBuffer(context,
			      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			      sizeof(cl_float) * WIDTH * HEIGHT, input, NULL);
  if (memobjs[1] == (cl_mem)0) 
    { 
      delete_memobjs(memobjs, 1);
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 

  program = clCreateProgramWithSource(context, 
				      1, (const char**)&source, NULL, NULL); 
  if (program == (cl_program)0) 
    { 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    }

  free (source);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  kernel = clCreateKernel(program, "matrix_transpose", NULL); 
  if (kernel == (cl_kernel)0) 
    { 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 

  err = clSetKernelArg(kernel,  0,  
		       sizeof(cl_mem), (void *) &memobjs[0]); 
  err |= clSetKernelArg(kernel, 1,  
			sizeof(cl_mem), (void *) &memobjs[1]); 
  err |= clSetKernelArg(kernel, 2,
			(32 + 1) * 32 * sizeof(float), NULL);
 
  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 

  global_work_size[0] = 2 * WIDTH; 
  global_work_size[1] = HEIGHT / 32; 
  local_work_size[0]= 64; 
  local_work_size[1]= 1; 

  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, 
			       global_work_size, local_work_size,  
			       0, NULL, NULL); 

  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  err = clEnqueueReadBuffer(cmd_queue, memobjs[0], CL_TRUE, 
			    0, WIDTH * (HEIGHT + PADDING) * sizeof(cl_float), output, 
			    0, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  delete_memobjs(memobjs, 2); 
  clReleaseKernel(kernel); 
  clReleaseProgram(program); 
  clReleaseCommandQueue(cmd_queue); 
  clReleaseContext(context); 

  for (i = 0; i < HEIGHT; ++i)
    {
      for (j = 0; j < WIDTH; ++j) {
	if (input[i * WIDTH + j] != output[j * (HEIGHT + PADDING) + i]) {
	  printf ("FAIL\n");
	  return -1;
	}
      }
    }
  
  printf ("OK\n");
  return 0;
}

static void 
delete_memobjs(cl_mem *memobjs, int n) 
{ 
  int i; 
  for (i=0; i<n; i++) 
    clReleaseMemObject(memobjs[i]); 
} 
