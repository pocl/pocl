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

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


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
  
  cl_platform_id platforms[1];
  cl_uint nplatforms;
  cl_device_id devices[1]; // + 1 for duplicate test
  cl_uint num_devices;

  cl_context context;
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

  err = clGetPlatformIDs(1, platforms, &nplatforms);	
  if (err != CL_SUCCESS && !nplatforms)
    return EXIT_FAILURE;
  
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1,
                       devices, &num_devices);  
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  context = clCreateContext(NULL, num_devices, devices, NULL, 
                                       NULL, &err);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id), devices, NULL);
  if (err != CL_SUCCESS) 
    {
      puts("clGetContextInfo call failed\n");
      goto error;
    }

  queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queue) 
    {
      puts("clCreateCommandQueue call failed\n");
      goto error;
    }

  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_a, &err);
  if (d_a == NULL)
    {
      printf("clCreateBuffer call failed err = %d\n", err);
      goto error;
    }
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_b, &err);
  if (d_b == NULL)
    {
      printf("clCreateBuffer call failed err = %d\n", err);
      goto error;
    }
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  if (d_c == NULL)
    {
      printf("clCreateBuffer call failed err = %d\n", err);
      goto error;
    }

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
  if (err != CL_SUCCESS) 
    {
      puts("clGetContextInfo call failed\n");
      goto error;
    }
 
  err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL );
  if (err != CL_SUCCESS) 
    {
      puts("clGetContextInfo call failed\n");
      goto error;
    }

  clFinish(queue);

  for(i = 0; i < n; i++)
    if(h_c[i] != 2 * i)
      {
        printf("Fail to validate vector\n");
        goto error;
      }
     
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
 
  free(h_a);
  free(h_b);
  free(h_c);

  return EXIT_SUCCESS;

error:
  return EXIT_FAILURE;
}
