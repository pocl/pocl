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

/* detects restrict, variadic macros etc */
#include "pocl_compiler_features.h"

#define POCL_FILENAME_LENGTH 1024

typedef struct _mem_mapping mem_mapping_t;
/* represents a single buffer to host memory mapping */
struct _mem_mapping {
  void *host_ptr; /* the location of the mapped buffer chunk in the host memory */
  size_t offset; /* offset to the beginning of the buffer */
  size_t size;
  mem_mapping_t *prev, *next;
  /* This is required, because two clEnqueueMap() with the same buffer+size+offset,
     will create two identical mappings in the buffer->mappings LL.
     Without this flag, both corresponding clEnqUnmap()s will find
     the same mapping (the first one in mappings LL), which will lead
     to memory double-free corruption later. */
  long unmap_requested;
};

typedef struct _mem_destructor_callback mem_destructor_callback_t;
/* represents a memory object destructor callback */
struct _mem_destructor_callback
{
  void (CL_CALLBACK * pfn_notify) (cl_mem, void*); /* callback function */
  void *user_data; /* user supplied data passed to callback function */
  mem_destructor_callback_t *next;
};

typedef struct _build_program_callback build_program_callback_t;
struct _build_program_callback
{
    void (CL_CALLBACK * callback_function) (cl_program, void*); /* callback function */
    void *user_data; /* user supplied data passed to callback function */
};

// Command Queue datatypes

// clEnqueueNDRangeKernel
typedef struct
{
  void *data;
  char *tmp_dir; 
  pocl_workgroup wg;
  cl_kernel kernel;
  size_t local_x;
  size_t local_y;
  size_t local_z;
  struct pocl_context pc;
  struct pocl_argument *arguments;
  /* Can be used to store/cache device-specific data. */
  void **device_data;
} _cl_command_run;

// clEnqueueNativeKernel
typedef struct
{
  void *data;
  void *args;
  size_t cb_args;
  void (*user_func)(void *);
  cl_mem *mem_list;
  unsigned num_mem_objects;
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
  cl_device_id src_dev;
  cl_device_id dst_dev;
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
  size_t h_origin[3];
  size_t region[3];
  size_t h_rowpitch;
  size_t h_slicepitch;
  size_t b_rowpitch;
  size_t b_slicepitch;
  cl_mem buffer;
} _cl_command_read_image;

typedef struct
{
  void *device_ptr;
  const void *host_ptr;
  size_t origin[3];
  size_t h_origin[3];
  size_t region[3];
  size_t h_rowpitch;
  size_t h_slicepitch;
  size_t b_rowpitch;
  size_t b_slicepitch;
  cl_mem buffer;
} _cl_command_write_image;

typedef struct
{
  cl_device_id dst_device;
  cl_mem dst_buffer;
  cl_device_id src_device;
  cl_mem src_buffer;
  size_t dst_origin[3];
  size_t src_origin[3];
  size_t region[3];
  size_t dst_rowpitch;
  size_t dst_slicepitch;
  size_t src_rowpitch;
  size_t src_slicepitch;
} _cl_command_copy_image;

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
  cl_mem buffer;
} _cl_command_fill_image;

/* clEnqueueFillBuffer */
typedef struct
{
  cl_mem buffer;
  void* ptr;
  size_t size, offset;
  void* pattern;
  size_t pattern_size;
} _cl_command_fill;

/* clEnqueueMarkerWithWaitlist */
typedef struct
{
  void *data;
  int has_wait_list;
} _cl_command_marker;

/* clEnqueueBarrierWithWaitlist */
typedef _cl_command_marker _cl_command_barrier;

/* clEnqueueMigrateMemObjects */
typedef struct
{
  void *data;
  size_t num_mem_objects;
  cl_mem *mem_objects;
  cl_device_id *source_devices;
} _cl_command_migrate;

typedef struct
{
  void* data;
  void* queue;
  unsigned  num_svm_pointers;
  void  **svm_pointers;
  void (CL_CALLBACK  *pfn_free_func) ( cl_command_queue queue,
                                       unsigned num_svm_pointers,
                                       void *svm_pointers[],
                                       void  *user_data);
} _cl_command_svm_free;

typedef struct
{
  void* svm_ptr;
  size_t size;
  cl_map_flags flags;
} _cl_command_svm_map;

typedef struct
{
  void* svm_ptr;
} _cl_command_svm_unmap;

typedef struct
{
  const void* src;
  void* dst;
  size_t size;
} _cl_command_svm_cpy;

typedef union
{
  _cl_command_run run;
  _cl_command_native native;
  _cl_command_read read;
  _cl_command_write write;
  _cl_command_copy copy;
  _cl_command_map map;
  _cl_command_fill_image fill_image;
  _cl_command_read_image read_image;
  _cl_command_write_image write_image;
  _cl_command_copy_image copy_image;
  _cl_command_marker marker;
  _cl_command_barrier barrier;
  _cl_command_unmap unmap;
  _cl_command_migrate migrate;
  _cl_command_fill memfill;

  _cl_command_svm_free svm_free;
  _cl_command_svm_map svm_map;
  _cl_command_svm_unmap svm_unmap;
  _cl_command_svm_cpy svm_memcpy;
} _cl_command_t;

// one item in the command queue
typedef struct _cl_command_node _cl_command_node;
struct _cl_command_node
{
  _cl_command_t command;
  cl_command_type type;
  _cl_command_node * volatile next; // for linked-list storage
  _cl_command_node * volatile prev;
  cl_event event;
  const cl_event *volatile event_wait_list;
  cl_device_id device;
  volatile cl_int ready;
};

#ifndef LLVM_7_0
#define LLVM_OLDER_THAN_7_0 1

#ifndef LLVM_6_0
#define LLVM_OLDER_THAN_6_0 1

#ifndef LLVM_5_0
#define LLVM_OLDER_THAN_5_0 1

#ifndef LLVM_4_0
#define LLVM_OLDER_THAN_4_0 1

#ifndef LLVM_3_9
#define LLVM_OLDER_THAN_3_9 1

#ifndef LLVM_3_8
#define LLVM_OLDER_THAN_3_8 1

#ifndef LLVM_3_7
#define LLVM_OLDER_THAN_3_7 1

#ifndef LLVM_3_6
#define LLVM_OLDER_THAN_3_6 1

#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif


#if (defined LLVM_4_0)
# define LLVM_OLDER_THAN_5_0 1
#endif

#endif /* POCL_H */
