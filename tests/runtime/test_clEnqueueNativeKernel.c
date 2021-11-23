/* Tests clEnqueueNativeKernel 

   Copyright (c) 2013 Kalray
   
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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "poclu.h"


struct native_kernel_args {
  double *a;
  double *b;
  double *c;
  int size;
};

static void
native_vec_add(void *data)
{
  struct native_kernel_args *args = (struct native_kernel_args *) data; 
  int i;

  for(i = 0; i < args->size; i++) {
    args->c[i] = args->a[i] + args->b[i];
  }

}

int main(int argc, char **argv) {
  unsigned int n = 100;
  
  double *h_a;
  double *h_b;
  double *h_c;
  cl_mem mem_list[3];
  const void *args_mem_loc[3];

  struct native_kernel_args args;
 
  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;
  
  cl_context ctx;
  cl_device_id did;
  cl_command_queue queue;
 
  size_t bytes = n * sizeof(double);
 
  h_a = (double *) malloc(bytes);
  h_b = (double *) malloc(bytes);
  h_c = (double *) malloc(bytes);
 
  size_t i;
  for( i = 0; i < n; i++ )
  {
    h_a[i] = (double)i;
    h_b[i] = (double)i;
  }

  cl_int err;

  CHECK_CL_ERROR(poclu_get_any_device(&ctx, &did, &queue));
  TEST_ASSERT( ctx );
  TEST_ASSERT( did );
  TEST_ASSERT( queue );

  d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_a);

  d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_b);

  d_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");
  TEST_ASSERT(d_c);

  args.size = n;
  args.a = 0;
  args.b = 0;
  args.c = 0;

  mem_list[0] = d_a;
  mem_list[1] = d_b;
  mem_list[2] = d_c;

  args_mem_loc[0] = &args.a;
  args_mem_loc[1] = &args.b;
  args_mem_loc[2] = &args.c;
  
  err = clEnqueueNativeKernel ( queue, native_vec_add, &args, sizeof(struct native_kernel_args),
          3, mem_list, args_mem_loc, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNativeKernel");
 
  err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL );
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");

  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish");

  for(i = 0; i < n; i++)
    if(h_c[i] != 2 * i)
      {
        printf("Fail to validate vector\n");
        goto error;
      }

  CHECK_CL_ERROR (clReleaseMemObject (d_a));
  CHECK_CL_ERROR (clReleaseMemObject (d_b));
  CHECK_CL_ERROR (clReleaseMemObject (d_c));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadCompiler ());

  free(h_a);
  free(h_b);
  free(h_c);

  printf ("OK\n");
  return EXIT_SUCCESS;

error:
  return EXIT_FAILURE;
}
