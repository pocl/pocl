/* pocl.h - global pocl declarations for the host side runtime.

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen

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

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 220
#endif
#include <CL/opencl.h>

#include "config.h"

#include "pocl_export.h"

#include "pocl_context.h"

/* detects restrict, variadic macros etc */
#include "pocl_compiler_features.h"

/* The maximum file, directory and path name lengths.
   NOTE: GDB seems to fail to load symbols from .so files which have
   longer pathnames than 511, thus the quite small dir/filename length
   limiter. */
#define POCL_MAX_DIRNAME_LENGTH 64
#define POCL_MAX_FILENAME_LENGTH (POCL_MAX_DIRNAME_LENGTH)
#define POCL_MAX_PATHNAME_LENGTH 4096

/* official Khronos ID */
#ifndef CL_KHRONOS_VENDOR_ID_POCL
#define CL_KHRONOS_VENDOR_ID_POCL 0x10006
#endif

/* represents a single buffer to host memory mapping */
typedef struct mem_mapping
{
  void *host_ptr; /* the location of the mapped buffer chunk in the host memory
                   */
  size_t offset;  /* offset to the beginning of the buffer */
  size_t size;
  struct mem_mapping *prev, *next;

  /* This is required, because two clEnqueueMap() with the same
     buffer+size+offset, will create two identical mappings in the
     buffer->mappings LL. Without this flag, both corresponding clEnqUnmap()s
     will find the same mapping (the first one in mappings LL), which will lead
     to memory double-free corruption later. */
  int unmap_requested;

  cl_map_flags map_flags;
  /* image mapping data */
  size_t origin[3];
  size_t region[3];
  size_t row_pitch;
  size_t slice_pitch;
} mem_mapping_t;

/* memory identifier:  */
typedef struct pocl_mem_identifier
{
  /* global-memory-specific pointer
     to hardware resource that represents memory. This may be anything, but
     must be non-NULL while the memory is actually allocated, and NULL when
     it's not */
  void *mem_ptr;

  /* If mem_ptr represents an address in the device global memory which
     is pinned for the lifetime of the buffer. */
  int is_pinned;

  /* The device-side memory address (if known). If is_pinned is true, this
     must be a valid value (note: 0 means invalid address). */
  void *device_addr;

  /* Content version tracking. Every write use (clEnqWriteBuffer,
   * clMapBuffer(CL_MAP_WRITE), write_only image, read/write buffers as kernel
   * args etc) increases the version; read uses do not. At command enqueue
   * time, the last version across all global mems AND mem_host_ptr
   * is found and migrated to the destination device's global mem.
   *
   * In theory, a simple bool of "valid/invalid could be used;
   * the only difference is that version saves history
   * (so we could in future do semi-intelligent memory GC, as in "i see the
   * buffer on this device hasn't been used for 100 versions, i can free it").
   */
  uint64_t version;

  /* Extra pointer for drivers to use for anything
   *
   * Currently CUDA uses it to track ALLOC_HOST_PTR allocations.
   * Vulkan uses it to store host-mapped staging memory
   */
  void *extra_ptr;

  /* Extra integer for drivers to use for anything
   *
   * Currently Vulkan uses it to track vulkan memory requirements
   */
  uint64_t extra;

} pocl_mem_identifier;

typedef char pixel_t[16];

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

// same as SHA1_DIGEST_SIZE
#define POCL_KERNEL_DIGEST_SIZE 20
typedef uint8_t pocl_kernel_hash_t[POCL_KERNEL_DIGEST_SIZE];

// clEnqueueNDRangeKernel
typedef struct
{
  void *hash;
  void *wg; /* The work group function ptr. Device specific. */
  cl_kernel kernel;
  /* The launch data that can be passed to the kernel execution environment. */
  struct pocl_context pc;
  struct pocl_argument *arguments;
  /* Can be used to store/cache arbitrary device-specific data. */
  void *device_data;
  /* If set to 1, disallow any work-group function specialization. */
  int force_generic_wg_func;
  /* If set to 1, disallow "small grid" WG function specialization. */
  int force_large_grid_wg_func;
} _cl_command_run;

// clEnqueueCommandBufferKHR
typedef struct
{
  cl_command_buffer_khr buffer;
} _cl_command_replay;

// clEnqueueNativeKernel
typedef struct
{
  void *args;
  size_t cb_args;
  void **arg_locs;
  void(CL_CALLBACK *user_func) (void *);
} _cl_command_native;

// clEnqueueReadBuffer
typedef struct
{
  void *__restrict__ dst_host_ptr;
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *src_content_size_mem_id;
  size_t offset;
  size_t size;
  size_t *content_size;
} _cl_command_read;

// clEnqueueWriteBuffer
typedef struct
{
  const void *__restrict__ src_host_ptr;
  pocl_mem_identifier *dst_mem_id;
  size_t offset;
  size_t size;
} _cl_command_write;

// clEnqueueCopyBuffer
typedef struct
{
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *dst_mem_id;
  pocl_mem_identifier *src_content_size_mem_id;
  cl_mem src;
  cl_mem src_content_size;
  cl_mem dst;
  size_t src_offset;
  size_t dst_offset;
  size_t size;
} _cl_command_copy;

// clEnqueueReadBufferRect
typedef struct
{
  void *__restrict__ dst_host_ptr;
  pocl_mem_identifier *src_mem_id;
  size_t buffer_origin[3];
  size_t host_origin[3];
  size_t region[3];
  size_t buffer_row_pitch;
  size_t buffer_slice_pitch;
  size_t host_row_pitch;
  size_t host_slice_pitch;
} _cl_command_read_rect;

// clEnqueueWriteBufferRect
typedef struct
{
  const void *__restrict__ src_host_ptr;
  pocl_mem_identifier *dst_mem_id;
  size_t buffer_origin[3];
  size_t host_origin[3];
  size_t region[3];
  size_t buffer_row_pitch;
  size_t buffer_slice_pitch;
  size_t host_row_pitch;
  size_t host_slice_pitch;
} _cl_command_write_rect;

// clEnqueueCopyBufferRect
typedef struct
{
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *dst_mem_id;
  cl_mem src;
  cl_mem dst;
  size_t dst_origin[3];
  size_t src_origin[3];
  size_t region[3];
  size_t src_row_pitch;
  size_t src_slice_pitch;
  size_t dst_row_pitch;
  size_t dst_slice_pitch;
} _cl_command_copy_rect;

// clEnqueueMapBuffer
typedef struct
{
  pocl_mem_identifier *mem_id;
  mem_mapping_t *mapping;
} _cl_command_map;

/* clEnqueueUnMapMemObject */
typedef struct
{
  pocl_mem_identifier *mem_id;
  mem_mapping_t *mapping;
} _cl_command_unmap;

/* clEnqueueFillBuffer */
typedef struct
{
  pocl_mem_identifier *dst_mem_id;
  size_t size;
  size_t offset;
  void *__restrict__ pattern;
  size_t pattern_size;
} _cl_command_fill_mem;

/* clEnqueue(Write/Read)Image */
typedef struct
{
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *dst_mem_id;
  void *__restrict__ dst_host_ptr;
  cl_mem src;
  cl_mem dst;
  size_t dst_offset;
  size_t origin[3];
  size_t region[3];
  size_t dst_row_pitch;
  size_t dst_slice_pitch;
} _cl_command_read_image;

typedef struct
{
  pocl_mem_identifier *dst_mem_id;
  pocl_mem_identifier *src_mem_id;
  const void *__restrict__ src_host_ptr;
  cl_mem src;
  cl_mem dst;
  size_t src_offset;
  size_t origin[3];
  size_t region[3];
  size_t src_row_pitch;
  size_t src_slice_pitch;
} _cl_command_write_image;

typedef struct
{
  pocl_mem_identifier *src_mem_id;
  pocl_mem_identifier *dst_mem_id;
  cl_mem src;
  cl_mem dst;
  size_t dst_origin[3];
  size_t src_origin[3];
  size_t region[3];
} _cl_command_copy_image;

/* clEnqueueFillImage */
typedef struct
{
  pixel_t fill_pixel;
  cl_uint4 orig_pixel;
  size_t pixel_size;
  pocl_mem_identifier *mem_id;
  size_t origin[3];
  size_t region[3];
} _cl_command_fill_image;

/* clEnqueueMarkerWithWaitlist */
typedef struct
{
  void *data;
  int has_wait_list;
} _cl_command_marker;

/* clEnqueueBarrierWithWaitlist */
typedef _cl_command_marker _cl_command_barrier;

typedef enum pocl_migration_type_e {
  ENQUEUE_MIGRATE_TYPE_NOP,
  ENQUEUE_MIGRATE_TYPE_D2H,
  ENQUEUE_MIGRATE_TYPE_H2D,
  ENQUEUE_MIGRATE_TYPE_D2D
} pocl_migration_type_t;

/* clEnqueueMigrateMemObjects */
typedef struct
{
  pocl_migration_type_t type;
  cl_device_id src_device;
  pocl_mem_identifier *src_id;
  pocl_mem_identifier *dst_id;
  pocl_mem_identifier *mem_id;
  /** For migrating a buffer that has a size buffer as per
   * cl_pocl_content_size */
  uint64_t migration_size;
  pocl_mem_identifier *src_content_size_mem_id;
} _cl_command_migrate;

typedef struct
{
  void* data;
  cl_command_queue queue;
  unsigned  num_svm_pointers;
  void  **svm_pointers;
  void (CL_CALLBACK  *pfn_free_func) ( cl_command_queue queue,
                                       unsigned num_svm_pointers,
                                       void *svm_pointers[],
                                       void  *user_data);
} _cl_command_svm_free;

typedef struct
{
  unsigned num_svm_pointers;
  size_t *sizes;
  void **svm_pointers;
} _cl_command_svm_migrate;

typedef struct
{
  void* svm_ptr;
  size_t size;
  cl_map_flags flags;
} _cl_command_svm_map;

typedef struct
{
  void* svm_ptr;
  size_t size;
} _cl_command_svm_unmap;

typedef struct
{
  const void *__restrict__ src;
  void *__restrict__ dst;
  size_t size;
} _cl_command_svm_cpy;

typedef struct
{
  const void *__restrict__ src;
  void *__restrict__ dst;
  size_t region[3];
  size_t src_origin[3];
  size_t dst_origin[3];
  size_t src_row_pitch;
  size_t src_slice_pitch;
  size_t dst_row_pitch;
  size_t dst_slice_pitch;
} _cl_command_svm_cpy_rect;

typedef struct
{
  void *__restrict__ svm_ptr;
  size_t size;
  void *__restrict__ pattern;
  size_t pattern_size;
} _cl_command_svm_fill;

typedef struct
{
  void *__restrict__ svm_ptr;
  void *__restrict__ pattern;
  size_t region[3];
  size_t origin[3];
  size_t row_pitch;
  size_t slice_pitch;
  size_t pattern_size;
} _cl_command_svm_fill_rect;

typedef struct
{
  const void *ptr;
  size_t size;
  cl_mem_advice_intel advice;
} _cl_command_svm_memadvise;

typedef union
{
  _cl_command_run run;
  _cl_command_native native;
  _cl_command_replay replay;

  _cl_command_read read;
  _cl_command_write write;
  _cl_command_copy copy;
  _cl_command_read_rect read_rect;
  _cl_command_write_rect write_rect;
  _cl_command_copy_rect copy_rect;
  _cl_command_fill_mem memfill;

  _cl_command_read_image read_image;
  _cl_command_write_image write_image;
  _cl_command_copy_image copy_image;
  _cl_command_fill_image fill_image;

  _cl_command_map map;
  _cl_command_unmap unmap;

  _cl_command_marker marker;
  _cl_command_barrier barrier;
  _cl_command_migrate migrate;

  _cl_command_svm_free svm_free;
  _cl_command_svm_map svm_map;
  _cl_command_svm_unmap svm_unmap;
  _cl_command_svm_cpy svm_memcpy;
  _cl_command_svm_cpy_rect svm_memcpy_rect;
  _cl_command_svm_fill svm_fill;
  _cl_command_svm_fill_rect svm_fill_rect;
  _cl_command_svm_migrate svm_migrate;

  _cl_command_svm_memadvise mem_advise;
} _cl_command_t;

typedef enum
{
  COMMAND_FAILED = -1,
  COMMAND_NOT_READY = 0,
  COMMAND_READY = 1,
} command_node_state;

// one item in the command queue or command buffer
typedef struct _cl_command_node _cl_command_node;
struct _cl_command_node {
  _cl_command_t command;
  cl_command_type type;
  _cl_command_node *next; // for linked-list storage
  _cl_command_node *prev;
  cl_int buffered;

  /***
   * Command buffers use sync points as a template for synchronizing commands
   * within the buffer. Commands outside the buffer can't depend on sync points
   * and individual commands in the buffer can't depend on events. Because this
   * struct is used both for recorded and immediately enqueued commands, the
   * two synchronization mechanisms are made mutually exclusive here.
   * */
  union
  {
    struct
    {
      cl_event event;
    } event;
    struct
    {
      cl_sync_point_khr sync_point;
      cl_uint num_sync_points_in_wait_list;
      cl_sync_point_khr *sync_point_wait_list;
    } syncpoint;
  } sync;
  cl_device_id device;
  /* The index of the targeted device in the **program** device list. */
  unsigned program_device_i;
  command_node_state node_state;

  /* fields needed by buffered commands only */

  /* Which of the command queues in the command buffer's queue list
   * this command was recorded for. */
  cl_uint queue_idx;
  /* List of buffers this command accesses, used for inserting migrations */
  cl_uint memobj_count;
  cl_mem *memobj_list;
  char *readonly_flag_list;
};

/**
 * Enumeration for different modes of converting automatic locals
 */
typedef enum
{
  POCL_AUTOLOCALS_TO_ARGS_NEVER = 0,
  POCL_AUTOLOCALS_TO_ARGS_ALWAYS = 1,
  /* convert autolocals to args only if there are dynamic local memory
   * function arguments in the kernel. */
  POCL_AUTOLOCALS_TO_ARGS_ONLY_IF_DYNAMIC_LOCALS_PRESENT = 2,
} pocl_autolocals_to_args_strategy;

#include "_libclang_versions_checks.h"

#endif /* POCL_H */
