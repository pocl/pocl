/* Tests a case where a program created from binary (with a loop) is run with
   a local size of (1, 1, 1).

   Copyright (c) 2017 pocl developers

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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define NB_WORK_GROUP 32
#define VEC_SIZE 32

const char *kernelSource =
  "__kernel void test(__global unsigned * restrict buffer,          "
  "                   __local unsigned * restrict local_input,      "
  "                   const unsigned vec_size)                      "
  "{                                                                "
  "  unsigned i, j;                                                 "
  "  size_t gid = get_global_id(0);                                 "
  "  size_t lid = get_local_id(0);                                  "
  "  size_t lsize = get_local_size(0);                              "
  "  event_t event_read, event_write;                               "
  "  event_read = async_work_group_copy(local_input, &buffer[gid*vec_size*lsize], vec_size*lsize, 0);"
  "  for (i=0; i<vec_size; i++)                                     "
  "    {                                                            "
  "      if (i == 0)                                                "
  "        wait_group_events(1, &event_read);                       "
  "      local_input[i*lsize+lid]++;                                "
  "    }                                                            "
  "  event_write = async_work_group_copy(&buffer[gid*vec_size*lsize], local_input, vec_size*lsize, event_write);"
  "  wait_group_events(1, &event_write);                            "
  "}                                                                ";

int main ()
{
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program_source, program_binary;
  cl_kernel kernel;
  cl_mem buffer;
  cl_uint vec_size = VEC_SIZE;
  cl_uint input_buffer[NB_WORK_GROUP * VEC_SIZE] = {0};
  cl_uint output_buffer[NB_WORK_GROUP * VEC_SIZE] = {0};
  cl_int err;

  size_t global_size = NB_WORK_GROUP;
  size_t local_size = 1;
  size_t sizeof_buffer = global_size * vec_size * sizeof(unsigned);
  size_t binary_size;

  char *binary;

  unsigned k, i;

  for (k=0; k<global_size; k++)
    {
      for (i=0; i<vec_size; i++)
        {
          input_buffer[k*vec_size+i]=k*vec_size+i;
        }
    }

  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_CL_ERROR (err);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
  CHECK_CL_ERROR (err);
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  CHECK_CL_ERROR (err);
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_CL_ERROR (err);

  program_source = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
  CHECK_CL_ERROR (err);
  err = clBuildProgram(program_source, 0, NULL, NULL, NULL, NULL);
  CHECK_CL_ERROR (err);
  err = clGetProgramInfo(program_source, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
  CHECK_CL_ERROR (err);
  binary = (char *)malloc(sizeof(char)*binary_size);
  TEST_ASSERT (binary);
  err = clGetProgramInfo(program_source, CL_PROGRAM_BINARIES, sizeof(char*), &binary, NULL);
  CHECK_CL_ERROR (err);

  program_binary = clCreateProgramWithBinary(context, 1, &device, &binary_size,
                                             (const unsigned char **)&binary, NULL, &err);
  CHECK_CL_ERROR (err);
  err = clBuildProgram(program_binary, 0, NULL, NULL, NULL, NULL);
  CHECK_CL_ERROR (err);
  kernel = clCreateKernel(program_binary, "test", &err);
  CHECK_CL_ERROR (err);
  buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof_buffer, &input_buffer, &err);
  CHECK_CL_ERROR (err);
  err = clSetKernelArg(kernel, 0, sizeof (cl_mem), &buffer);
  CHECK_CL_ERROR (err);
  err = clSetKernelArg(kernel, 1, sizeof (unsigned) * vec_size, NULL);
  CHECK_CL_ERROR (err);
  err = clSetKernelArg(kernel, 2, sizeof (cl_uint), &vec_size);
  CHECK_CL_ERROR (err);
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  CHECK_CL_ERROR (err);
  clFinish(queue);
  err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof_buffer, output_buffer, 0, NULL, NULL);
  CHECK_CL_ERROR (err);

  for (k=0; k<global_size; k++)
    {
      for (i=0; i<vec_size; i++)
        {
          unsigned expected = (input_buffer[k*vec_size+i]+1);
          if (output_buffer[k*vec_size+i] != expected)
            {
              printf("Error at %u %u : %u != %u\n", k, i,
                     output_buffer[k*vec_size+i], expected);
              return 1;
            }
        }
    }

  clReleaseMemObject(buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program_source);
  clReleaseProgram(program_binary);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseDevice(device);
  clUnloadPlatformCompiler(platform);
  return 0;
}
