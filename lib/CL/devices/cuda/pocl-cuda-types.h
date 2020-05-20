/* pocl-cuda.c - driver for CUDA devices

   Copyright (c) 2016-2017 James Price / University of Bristol

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal
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
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "config.h"

#include "common.h"
#include "pocl.h"
#include "pocl_util.h"

#include <cuda.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct pocl_cuda_device_data_s
{
  CUdevice device;
  CUcontext context;
  CUevent epoch_event;
  cl_ulong epoch;
  char libdevice[PATH_MAX];
  pocl_lock_t compile_lock;
} pocl_cuda_device_data_t;

typedef struct pocl_cuda_queue_data_s
{
  CUstream stream;
  int use_threads;
  pthread_t submit_thread;
  pthread_t finalize_thread;
  pthread_mutex_t lock;
  pthread_cond_t pending_cond;
  pthread_cond_t running_cond;
  _cl_command_node *volatile pending_queue;
  _cl_command_node *volatile running_queue;
  cl_command_queue queue;
} pocl_cuda_queue_data_t;

typedef struct pocl_cuda_kernel_data_s
{
  CUmodule module;
  CUmodule module_offsets;
  CUfunction kernel;
  CUfunction kernel_offsets;
  size_t *alignments;
  size_t auto_local_offset;
} pocl_cuda_kernel_data_t;

typedef struct pocl_cuda_event_data_s
{
  CUevent start;
  CUevent end;
  volatile int events_ready;
  cl_int *ext_event_flag;
  volatile unsigned num_ext_events;
} pocl_cuda_event_data_t;
