/* pocl.h - global pocl declarations.

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

#ifndef POCL_H
#define POCL_H

#include <CL/opencl.h>

#define POCL_ADDRESS_SPACE_PRIVATE 0
#define POCL_ADDRESS_SPACE_GLOBAL 3
#define POCL_ADDRESS_SPACE_LOCAL 4
#define POCL_ADDRESS_SPACE_CONSTANT 5

struct pocl_context {
  cl_uint work_dim;
  size_t num_groups[3];
  size_t group_id[3];
  size_t global_offset[3];
};

typedef void (*pocl_workgroup) (void **, struct pocl_context *);


// Command Queue datatypes

// clEnqueueNDRangeKernel
typedef struct
{
  void *data;
  const char *file; 
  cl_kernel kernel;
  /* A list of argument buffers to free after the command has 
     been executed. */
  cl_mem *arg_buffers;
  int arg_buffer_count;
  struct pocl_context pc;
} _cl_command_run;

// clEnqueueReadBuffer
typedef struct
{
  void *data;
  void *host_ptr;
  const void *device_ptr;
  size_t cb;
  cl_mem buffer;
} _cl_command_read;

// clEnqueueWriteBuffer
typedef struct
{
  void *data;
  const void *host_ptr;
  void *device_ptr;
  size_t cb;
  cl_mem buffer;
} _cl_command_write;

// clEnqueueCopyBuffer
typedef struct
{
  void *data;
  void *src_ptr;
  void *dst_ptr;
  size_t cb;
  cl_mem src_buffer;
  cl_mem dst_buffer;
} _cl_command_copy;

typedef union
{
  _cl_command_run run;
  _cl_command_read read;
  _cl_command_write write;
  _cl_command_copy copy;
} _cl_command_t;

typedef enum
{
  CL_COMMAND_TYPE_READ,
  CL_COMMAND_TYPE_WRITE,
  CL_COMMAND_TYPE_RUN,
  CL_COMMAND_TYPE_COPY
} _cl_command_en;

// one item in the command queue
typedef struct
{
  _cl_command_t command;
  _cl_command_en type;
  void *next; // for linked-list storage
} _cl_command_node;

#endif /* POCL_H */
