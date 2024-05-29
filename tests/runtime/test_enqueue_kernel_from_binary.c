/* Tests a kernel create with binary.

   Copyright (c) 2016 pocl developers

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
#include <string.h>

#define BUFFER_SIZE 1024

// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                               "\n"
  "__kernel void vecAdd( __constant int *a,                                 \n"
  "                      __constant int *b,                                 \n"
  "                      __global int *c)                                   \n"
  "{                                                                        \n"
  "unsigned int i = get_global_id(0) * get_global_size(1)                   \n"
  "    + get_global_id(1);                                                  \n"
  "c[i] = a[i] + b[i] + get_local_id(0) * get_local_size(1)                 \n"
  "    + get_local_id(1);                                                   \n"
  "}                                                                        \n";

const char *barrier_kernelSource =
  "__kernel void barrier_kernel (global int *buffer) \n"
  "{ \n"
  "  private int a = buffer[get_global_id(0)-1] \n"
  "    + buffer[get_global_id(0)] + buffer[get_global_id(0)+1]; \n"
  "  barrier(CLK_LOCAL_MEM_FENCE); \n"
  "  buffer[get_global_id(0)] = a/3; \n"
  "} \n";

const char *barrier_kernel_reqd_wg_size_source =
  "__attribute__((reqd_work_group_size(6, 1, 1)))\n"
  "__kernel void static_wg_kernel (global int *buffer) \n"
  "{ \n"
  "  buffer[get_local_id (0)] = 2 + get_global_id (0); \n"
  "} \n";

int main(void)
{
  int *h_a;
  int *h_b;
  int *h_c1;
  int *h_c2;
  int *h_c3;
  int *bb1;
  int *bb2;
  int *static_wg_buf;

  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c1;
  cl_mem d_c2;
  cl_mem d_c3;
  cl_mem barrier_buffer1;
  cl_mem barrier_buffer2;
  cl_mem static_wg_buffer;

  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id device_id = NULL;
  cl_command_queue queue = NULL;
  cl_program program1, program2;
  cl_program b_program1, b_program2, static_wg_size_program,
    static_wg_size_bin_program;
  cl_kernel kernel1, kernel2, kernel3, barrier_kernel1, barrier_kernel2,
    static_wg_kernel;

  size_t globalSize[2], localSize[2];
  cl_int err;

//###################################################################
//###################################################################
// Initialize variables

  unsigned bytes = BUFFER_SIZE * sizeof(int);

  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);
  h_c1 = (int*)malloc(bytes);
  h_c2 = (int*)malloc(bytes);
  h_c3 = (int*)malloc(bytes);
  bb1 = (int*)malloc(bytes);
  bb2 = (int*)malloc(bytes);
  static_wg_buf = (int*)malloc(bytes);

  unsigned int i;
  for( i = 0; i < BUFFER_SIZE; i++ )
  {
    h_a[i] = 2*i-1;
    h_b[i] = -i;
  }

  for (i = 0; i < 8; ++i)
    {
      bb1[i] = bb2[i] = i*3;
    }
  memset(h_c1, 0, bytes);
  memset(h_c2, 0, bytes);
  memset(h_c3, 0, bytes);

  /* Initialize cl_programs.  */

  CHECK_CL_ERROR (
    poclu_get_any_device2 (&context, &device_id, &queue, &platform));
  TEST_ASSERT( context );
  TEST_ASSERT( device_id );
  TEST_ASSERT( queue );

  program1 = clCreateProgramWithSource(context, 1,
        (const char **) & kernelSource, NULL, &err);
  TEST_ASSERT(program1);

  CHECK_CL_ERROR(clBuildProgram(program1, 0, NULL, NULL, NULL, NULL));

  size_t binary_sizes;
  unsigned char *binary;
  CHECK_CL_ERROR(clGetProgramInfo(program1, CL_PROGRAM_BINARY_SIZES,
                         sizeof(size_t), &binary_sizes, NULL));

  binary = malloc(sizeof(unsigned char)*binary_sizes);
  TEST_ASSERT (binary);

  CHECK_CL_ERROR (clGetProgramInfo (program1, CL_PROGRAM_BINARIES,
                                    sizeof(unsigned char*), &binary, NULL));

  program2
    = clCreateProgramWithBinary (context, 1, &device_id, &binary_sizes,
                                 (const unsigned char **)(&binary), NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithBinary");

  CHECK_CL_ERROR (clBuildProgram (program2, 0, NULL, NULL, NULL, NULL));

  /* Barrier programs.  */
  b_program1 = clCreateProgramWithSource (context, 1,
                                          (const char **)
                                          &barrier_kernelSource,
                                          NULL, &err);

  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");
  TEST_ASSERT(b_program1);
  CHECK_CL_ERROR(clBuildProgram(b_program1, 0, NULL, NULL, NULL, NULL));

  static_wg_size_program
    = clCreateProgramWithSource (context, 1,
                                 (const char **)
                                 &barrier_kernel_reqd_wg_size_source,
                                 NULL, &err);

  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  TEST_ASSERT (static_wg_size_program);

  CHECK_CL_ERROR (clBuildProgram (static_wg_size_program, 0, NULL, NULL, NULL,
                                  NULL));

  size_t barrier_binary_sizes;
  unsigned char *barrier_binary;
  CHECK_CL_ERROR(clGetProgramInfo(b_program1, CL_PROGRAM_BINARY_SIZES,
                         sizeof(size_t), &barrier_binary_sizes, NULL));

  barrier_binary = malloc(sizeof(unsigned char)*barrier_binary_sizes);
  TEST_ASSERT (barrier_binary);

  CHECK_CL_ERROR(clGetProgramInfo(b_program1, CL_PROGRAM_BINARIES,
                         sizeof(unsigned char*), &barrier_binary, NULL));
  b_program2 =
    clCreateProgramWithBinary(context, 1, &device_id,
                              &barrier_binary_sizes,
                              (const unsigned char **)(&barrier_binary),
                              NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithBinary 2");

  CHECK_CL_ERROR(clBuildProgram(b_program2, 0, NULL, NULL, NULL, NULL));

  size_t static_wg_binary_sizes;
  unsigned char *static_wg_binary;
  CHECK_CL_ERROR (clGetProgramInfo (static_wg_size_program, CL_PROGRAM_BINARY_SIZES,
                                    sizeof (size_t), &static_wg_binary_sizes,
                                    NULL));

  static_wg_binary = malloc (sizeof (unsigned char)*static_wg_binary_sizes);
  TEST_ASSERT (static_wg_binary);

  CHECK_CL_ERROR (clGetProgramInfo (static_wg_size_program, CL_PROGRAM_BINARIES,
                                    sizeof (unsigned char*), &static_wg_binary,
                                    NULL));

  static_wg_size_bin_program =
    clCreateProgramWithBinary (context, 1, &device_id,
                               &static_wg_binary_sizes,
                               (const unsigned char **)(&static_wg_binary),
                               NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithBinary (static wg size)");

  CHECK_CL_ERROR (clBuildProgram (static_wg_size_bin_program, 0, NULL, NULL, NULL,
                                  NULL));

  /* Set up buffers in memory.  */

  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_a);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_b);

  d_c1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_c1);
  d_c2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_c2);
  d_c3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_c3);

  barrier_buffer1 = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bytes, bb1,
                                   NULL);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(barrier_buffer1);

  barrier_buffer2 = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bytes, bb2,
                                   NULL);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(barrier_buffer2);

  static_wg_buffer = clCreateBuffer (context, CL_MEM_COPY_HOST_PTR, bytes,
                                     static_wg_buf, NULL);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  TEST_ASSERT (static_wg_buffer);

  CHECK_CL_ERROR(clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                      bytes, h_a, 0, NULL, NULL));
  CHECK_CL_ERROR(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                      bytes, h_b, 0, NULL, NULL));

  /* Create kernels.  */

  kernel1 = clCreateKernel(program1, "vecAdd", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");
  TEST_ASSERT(kernel1);

  CHECK_CL_ERROR(clSetKernelArg(kernel1, 0, sizeof(cl_mem), &d_a));
  CHECK_CL_ERROR(clSetKernelArg(kernel1, 1, sizeof(cl_mem), &d_b));
  CHECK_CL_ERROR(clSetKernelArg(kernel1, 2, sizeof(cl_mem), &d_c1));

  barrier_kernel1 = clCreateKernel(b_program1, "barrier_kernel", &err);
  TEST_ASSERT(barrier_kernel1);

  CHECK_CL_ERROR(clSetKernelArg(barrier_kernel1, 0, sizeof(cl_mem), &barrier_buffer1));

  kernel2 = clCreateKernel(program2, "vecAdd", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");
  TEST_ASSERT(kernel2);

  CHECK_CL_ERROR(clSetKernelArg(kernel2, 0, sizeof(cl_mem), &d_a));
  CHECK_CL_ERROR(clSetKernelArg(kernel2, 1, sizeof(cl_mem), &d_b));
  CHECK_CL_ERROR(clSetKernelArg(kernel2, 2, sizeof(cl_mem), &d_c2));

  barrier_kernel2 = clCreateKernel(b_program2, "barrier_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");
  TEST_ASSERT(barrier_kernel2);

  CHECK_CL_ERROR(clSetKernelArg(barrier_kernel2, 0, sizeof(cl_mem), &barrier_buffer2));

  kernel3 = clCreateKernel(program2, "vecAdd", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");
  TEST_ASSERT(kernel3);

  CHECK_CL_ERROR(clSetKernelArg(kernel3, 0, sizeof(cl_mem), &d_a));
  CHECK_CL_ERROR(clSetKernelArg(kernel3, 1, sizeof(cl_mem), &d_b));
  CHECK_CL_ERROR(clSetKernelArg(kernel3, 2, sizeof(cl_mem), &d_c3));

  static_wg_kernel
    = clCreateKernel (static_wg_size_bin_program, "static_wg_kernel", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");
  TEST_ASSERT (static_wg_kernel);

  CHECK_CL_ERROR (clSetKernelArg (static_wg_kernel, 0, sizeof (cl_mem),
                                  &static_wg_buffer));

  /* Enqueue kernels.  */

  localSize[0] = 4;
  localSize[1] = 8;
  globalSize[0] = 16;
  globalSize[1] = 64;
  CHECK_CL_ERROR(clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, globalSize, localSize,
                               0, NULL, NULL));

  CHECK_CL_ERROR(clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, globalSize, localSize,
                               0, NULL, NULL));

  localSize[0] = 2;
  localSize[1] = 8;
  globalSize[0] = 16;
  globalSize[1] = 64;
  CHECK_CL_ERROR(clEnqueueNDRangeKernel(queue, kernel3, 2, NULL, globalSize, localSize,
                               0, NULL, NULL));
  TEST_ASSERT (!err);

  localSize[0] = 6;
  globalSize[0] = 6;
  size_t global_offset = 1;
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, barrier_kernel1, 1,
                                          &global_offset,
                                          globalSize,
                                          localSize,
                                          0, NULL, NULL));

  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, barrier_kernel2,
                                          1, &global_offset,
                                          globalSize,
                                          localSize,
                                          0, NULL, NULL));

  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, static_wg_kernel, 1,
                                          &global_offset,
                                          globalSize,
                                          localSize,
                                          0, NULL, NULL));

  /* Read output buffers.  */

  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, d_c1, CL_TRUE, 0,
                                       bytes, h_c1, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, d_c2, CL_TRUE, 0,
                                       bytes, h_c2, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, d_c3, CL_TRUE, 0,
                                       bytes, h_c3, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, barrier_buffer1, CL_TRUE, 0,
                                       bytes, bb1, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, barrier_buffer2, CL_TRUE, 0,
                                       bytes, bb2, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueReadBuffer (queue, static_wg_buffer, CL_TRUE, 0,
                                       bytes, static_wg_buf, 0, NULL, NULL));
  CHECK_CL_ERROR (clFinish (queue));


  /* Check output of each kernels.  */

  unsigned errors = 0;
  for(i = 0; i < BUFFER_SIZE; i++)
  {
    if(h_c1[i] != h_c2[i])
    {
      printf("[1] Check failed at offset %d, %i instead of %i\n", i, h_c2[i], h_c1[i]);
      if (++errors > 10)
        return EXIT_FAILURE;
    }
    if ((((i/128)%2) && (h_c1[i]-16 != h_c3[i])))
    {
      printf("[2] Check failed at offset %d, %i instead of %i\n", i, h_c3[i], h_c1[i]-16);
      if (++errors > 10)
        return EXIT_FAILURE;
    }
    if (!((i/128)%2) && (h_c1[i] != h_c3[i]))
    {
      printf("[3] Check failed at offset %d, %i instead of %i\n", i, h_c3[i], h_c1[i]);
      if (++errors > 10)
        return EXIT_FAILURE;
    }
  }

  for (i = 0; i < 6; ++i)
    {
      if (bb1[i] != bb2[i])
        {
          printf("barrier kernel failed at index %d: (%d != %d)\n", i, bb1[i],
                 bb2[i]);
          return EXIT_FAILURE;
        }
      if (static_wg_buf[i] != 2 + (int)i + 1)
        {
          printf("static wg kernel failed at index %d (%d != %d)\n", i,
                 static_wg_buf[i], 2 + (int)i + 1);
          return EXIT_FAILURE;
        }
    }

  CHECK_CL_ERROR (clReleaseMemObject (d_a));
  CHECK_CL_ERROR (clReleaseMemObject (d_b));
  CHECK_CL_ERROR (clReleaseMemObject (d_c1));
  CHECK_CL_ERROR (clReleaseMemObject (d_c2));
  CHECK_CL_ERROR (clReleaseMemObject (d_c3));
  CHECK_CL_ERROR (clReleaseMemObject (barrier_buffer1));
  CHECK_CL_ERROR (clReleaseMemObject (barrier_buffer2));
  CHECK_CL_ERROR (clReleaseMemObject (static_wg_buffer));

  CHECK_CL_ERROR (clReleaseKernel (kernel1));
  CHECK_CL_ERROR (clReleaseKernel (kernel2));
  CHECK_CL_ERROR (clReleaseKernel (kernel3));
  CHECK_CL_ERROR (clReleaseKernel (barrier_kernel1));
  CHECK_CL_ERROR (clReleaseKernel (barrier_kernel2));
  CHECK_CL_ERROR (clReleaseKernel (static_wg_kernel));

  CHECK_CL_ERROR (clReleaseProgram (program1));
  CHECK_CL_ERROR (clReleaseProgram (program2));
  CHECK_CL_ERROR (clReleaseProgram (b_program1));
  CHECK_CL_ERROR (clReleaseProgram (b_program2));
  CHECK_CL_ERROR (clReleaseProgram (static_wg_size_bin_program));
  CHECK_CL_ERROR (clReleaseProgram (static_wg_size_program));

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  free(bb1);
  free(bb2);
  free(h_a);
  free(h_b);
  free(h_c1);
  free(h_c2);
  free(h_c3);
  free(binary);
  free(barrier_binary);
  free (static_wg_binary);
  free (static_wg_buf);

  if (errors)
    return EXIT_FAILURE;
  printf ("OK\n");
  return EXIT_SUCCESS;
}
