/* pocl.h - global pocl declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2013 Pekka Jääskeläinen / Tampere University of Technology
  
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

/**
 * @file pocl.h
 * 
 * The declarations in this file are such that are used both in the
 * libpocl implementation CL and the kernel compiler. Others should be
 * moved to pocl_cl.h of lib/CL or under the kernel compiler dir. 
 * @todo Check if there are extra declarations here that could be moved.
 */
#ifndef POCL_H
#define POCL_H

#include <CL/opencl.h>

#include "pocl_device.h"
#include "config.h"

/*
 * During pocl kernel compiler transformations we use the fixed address 
 * space ids of clang's -ffake-address-space-map to mark the different 
 * address spaces to keep the processing target-independent. These
 * are converted to the target's address space map (if any), in a final
 * kernel compiler pass.
 */
#define POCL_ADDRESS_SPACE_PRIVATE 0
#define POCL_ADDRESS_SPACE_GLOBAL 1
#define POCL_ADDRESS_SPACE_LOCAL 2
#define POCL_ADDRESS_SPACE_CONSTANT 3

typedef struct _mem_mapping mem_mapping_t;
/* represents a single buffer to host memory mapping */
struct _mem_mapping {
  void *host_ptr; /* the location of the mapped buffer chunk in the host memory */
  size_t offset; /* offset to the beginning of the buffer */
  size_t size;
  mem_mapping_t *prev, *next;
};

// Command Queue datatypes

// clEnqueueNDRangeKernel
typedef struct
{
  void *data;
  char *tmp_dir; 
  pocl_workgroup wg;
  cl_kernel kernel;
  /* A list of argument buffers to free after the command has 
     been executed. */
  cl_mem *arg_buffers;
  int arg_buffer_count;
  size_t local_x;
  size_t local_y;
  size_t local_z;
  struct pocl_context pc;
  struct pocl_argument *arguments;
} _cl_command_run;

// clEnqueueNativeKernel
typedef struct
{
  void *data;
  void *args;
  size_t cb_args;
  void (*user_func)(void *);
  cl_mem *mem_list;
  int num_mem_objects;
} _cl_command_native;

// clEnqueueReadBuffer
typedef struct
{
  void *host_ptr;
  const void *device_ptr;
  size_t offset;
  size_t cb;
  cl_mem buffer;
} _cl_command_read;

// clEnqueueWriteBuffer
typedef struct
{
  const void *host_ptr;
  void *device_ptr;
  size_t offset;
  size_t cb;
  cl_mem buffer;
} _cl_command_write;

// clEnqueueCopyBuffer
typedef struct
{
  void *data;
  void *src_ptr;
  size_t src_offset;
  void *dst_ptr;
  size_t dst_offset;
  size_t cb;
  cl_mem src_buffer;
  cl_mem dst_buffer;
} _cl_command_copy;

// clEnqueueMapBuffer
typedef struct
{
  cl_mem buffer;
  mem_mapping_t *mapping;
} _cl_command_map;


/* clEnqueue(Write/Read)Image */
typedef struct
{
  void *device_ptr;
  void *host_ptr;
  size_t origin[3];
  size_t region[3];
  size_t rowpitch;
  size_t slicepitch;
  cl_mem buffer;
} _cl_command_rw_image;

/* clEnqueueUnMapMemObject */
typedef struct
{
  void *data;
  cl_mem memobj;
  mem_mapping_t *mapping;
} _cl_command_unmap;

/* clEnqueueFillImage */
typedef struct
{
  void *data;
  void *device_ptr;
  size_t buffer_origin[3];
  size_t region[3];
  size_t rowpitch;
  size_t slicepitch;
  void *fill_pixel;
  size_t pixel_size;
} _cl_command_fill_image;

typedef struct
{
  void *data;
} _cl_command_marker;

typedef union
{
  _cl_command_run run;
  _cl_command_native native;
  _cl_command_read read;
  _cl_command_write write;
  _cl_command_copy copy;
  _cl_command_map map;
  _cl_command_fill_image fill_image;
  _cl_command_rw_image rw_image;
  _cl_command_marker marker;
  _cl_command_unmap unmap;
} _cl_command_t;

// one item in the command queue
typedef struct _cl_command_node_struct
{
  _cl_command_t command;
  cl_command_type type;
  struct _cl_command_node_struct *next; // for linked-list storage
  cl_event event;
  const cl_event *event_wait_list;
  cl_int num_events_in_wait_list;
  cl_device_id device;
} _cl_command_node;

/* Additional LLVM version macros to simplify ifdefs */

#if defined(LLVM_3_2) || defined(LLVM_3_3) || defined(LLVM_3_4) || \
    defined(LLVM_3_5)

# define LLVM_OLDER_THAN_3_6 1

#endif

#endif /* POCL_H */
