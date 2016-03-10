/* pocl-cuda.c - driver for CUDA devices

   Copyright (c) 2016 James Price / University of Bristol

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

#include "config.h"

#include "pocl-cuda.h"
#include "common.h"
#include "devices.h"

#include <cuda.h>

static void pocl_cuda_abort_on_error(CUresult result,
                                     unsigned line,
                                     const char* func,
                                     const char* code,
                                     const char* api)
{
  if (result != CUDA_SUCCESS)
  {
    const char *err_name;
    const char *err_string;
    cuGetErrorName(result, &err_name);
    cuGetErrorString(result, &err_string);
    POCL_MSG_PRINT2(func, line, "-> Error during %s\n", api);
    POCL_ABORT("-> %s: %s\n", err_name, err_string);
  }
}

#define CUDA_CHECK(result, api) \
  pocl_cuda_abort_on_error(result, __LINE__, __FUNCTION__, #result, api);


void
pocl_cuda_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops (ops);

  ops->device_name = "CUDA";
  ops->init_device_infos = pocl_cuda_init_device_infos;
  ops->probe = pocl_cuda_probe;
  ops->uninit = pocl_cuda_uninit;
  ops->init = pocl_cuda_init;
  //ops->alloc_mem_obj = pocl_cuda_alloc_mem_obj;
  //ops->free = pocl_cuda_free;
  //ops->compile_submitted_kernels = pocl_cuda_compile_submitted_kernels;
  //ops->run = pocl_cuda_run;
  //ops->read = pocl_basic_read;
  //ops->read_rect = pocl_basic_read_rect;
  //ops->write = pocl_basic_write;
  //ops->write_rect = pocl_basic_write_rect;
  //ops->copy = pocl_cuda_copy;
  //ops->copy_rect = pocl_basic_copy_rect;
  //ops->get_timer_value = pocl_cuda_get_timer_value;
}

void
pocl_cuda_init(cl_device_id device, const char* parameters)
{
  CUresult result;

  result = cuInit(0);
  CUDA_CHECK(result, "cuInit");

  if (device->data)
    return;

  device->data = malloc(sizeof(CUdevice));
  CUdevice *cuda_device = (CUdevice*)device->data;
  result = cuDeviceGet(cuda_device, 0);
  CUDA_CHECK(result, "cuDeviceGet");

  // Get specific device name
  device->long_name = device->short_name = malloc(256*sizeof(char));
  cuDeviceGetName(device->long_name, 256, *cuda_device);

  // Get other device properties
  cuDeviceGetAttribute((int*)&device->max_compute_units,
                       CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                       *cuda_device);
}

void
pocl_cuda_init_device_infos(struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos(dev);

  dev->type = CL_DEVICE_TYPE_GPU;
}

unsigned int
pocl_cuda_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  // TODO: Check how many CUDA device available (if any)

  if(env_count < 0)
    return 1;

  return env_count;
}

void
pocl_cuda_uninit(cl_device_id device)
{
  POCL_MEM_FREE(device->data);
  device->data = NULL;

  POCL_MEM_FREE(device->long_name);
}
