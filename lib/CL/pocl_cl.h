/* pocl_cl.h - local runtime library declarations.

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

#ifndef POCL_CL_H
#define POCL_CL_H

#include "config.h"
#include <stdio.h>
#include <ltdl.h>
#include <pthread.h>
#include "pocl.h"

#define POCL_FILENAME_LENGTH 1024

#define POCL_BUILD "pocl-build"
#define POCL_KERNEL "pocl-kernel"
#define POCL_WORKGROUP "pocl-workgroup"

#define POCL_ERROR(x) do { if (errcode_ret != NULL) {*errcode_ret = (x); return NULL;} } while (0)

#define POCL_PRINT_RUNTIME_WARNING(x) fprintf(stderr, "pocl warning: " x)

typedef pthread_mutex_t pocl_lock_t;

/* Generic functionality for handling different types of 
   OpenCL (host) objects. */

#define POCL_LOCK(__LOCK__) pthread_mutex_lock (&__LOCK__)
#define POCL_UNLOCK(__LOCK__) pthread_mutex_unlock (&__LOCK__)
#define POCL_INIT_LOCK(__LOCK__) pthread_mutex_init (&__LOCK__, NULL)

#define POCL_LOCK_OBJ(__OBJ__) POCL_LOCK(__OBJ__->pocl_lock)
#define POCL_UNLOCK_OBJ(__OBJ__) POCL_UNLOCK(__OBJ__->pocl_lock)

#define POCL_RELEASE_OBJECT(__OBJ__)             \
  do {                                           \
    POCL_LOCK_OBJ (__OBJ__);                     \
    __OBJ__->pocl_refcount--;                    \
    POCL_UNLOCK_OBJ (__OBJ__);                   \
  } while (0)                          

#define POCL_RETAIN_OBJECT(__OBJ__)             \
  do {                                          \
    POCL_LOCK_OBJ (__OBJ__);                    \
    __OBJ__->pocl_refcount++;                   \
    POCL_UNLOCK_OBJ (__OBJ__);                  \
  } while (0)

/* The reference counter is initialized to 1,
   when it goes to 0 object can be freed. */
#define POCL_INIT_OBJECT(__OBJ__)                \
  do {                                           \
    POCL_INIT_LOCK (__OBJ__->pocl_lock);         \
    __OBJ__->pocl_refcount = 1;                  \
  } while (0)

/* Declares the generic pocl object attributes inside a struct. */
#define POCL_OBJECT \
  pocl_lock_t pocl_lock; \
  int pocl_refcount 

struct pocl_argument {
  size_t size;
  void *value;
};

struct _cl_device_id {
  /* queries */
  cl_device_type type;
  cl_uint vendor_id;
  cl_uint max_compute_units;
  cl_uint max_work_item_dimensions;
  size_t *max_work_item_sizes;
  size_t max_work_group_size;
  cl_uint preferred_vector_width_char;
  cl_uint preferred_vector_width_short;
  cl_uint preferred_vector_width_int;
  cl_uint preferred_vector_width_long;
  cl_uint preferred_vector_width_float;
  cl_uint preferred_vector_width_double;
  cl_uint max_clock_frequency;
  cl_uint address_bits;
  cl_ulong max_mem_alloc_size;
  cl_bool image_support;
  cl_uint max_read_image_args;
  cl_uint max_write_image_args;
  size_t image2d_max_width;
  size_t image2d_max_height;
  size_t image3d_max_width;
  size_t image3d_max_height;
  size_t image3d_max_depth;
  cl_uint max_samplers;
  size_t max_parameter_size;
  cl_uint mem_base_addr_align;
  cl_uint min_data_type_align_size;
  cl_device_fp_config single_fp_config;
  cl_device_mem_cache_type global_mem_cache_type;
  cl_uint global_mem_cacheline_size;
  cl_ulong global_mem_cache_size;
  cl_ulong global_mem_size;
  cl_ulong max_constant_buffer_size;
  cl_uint max_constant_args;
  cl_device_local_mem_type local_mem_type;
  cl_ulong local_mem_size;
  cl_bool error_correction_support;
  size_t profiling_timer_resolution;
  cl_bool endian_little;
  cl_bool available;
  cl_bool compiler_available;
  cl_device_exec_capabilities execution_capabilities;
  cl_command_queue_properties queue_properties;
  cl_platform_id platform;
  char *name;
  char *vendor;
  char *driver_version;
  char *profile;
  char *version;
  char *extensions;
  /* implementation */
  void (*uninit) (cl_device_id device);
  void (*init) (cl_device_id device);
  void *(*malloc) (void *data, cl_mem_flags flags,
		   size_t size, void *host_ptr);
  void (*free) (void *data, cl_mem_flags flags, void *ptr);
  void (*read) (void *data, void *host_ptr, void *device_ptr, size_t cb);
  void (*read_rect) (void *data, void *host_ptr, void *device_ptr,
                     const size_t *buffer_origin,
                     const size_t *host_origin, 
                     const size_t *region,
                     size_t buffer_row_pitch,
                     size_t buffer_slice_pitch,
                     size_t host_row_pitch,
                     size_t host_slice_pitch);
  void (*write) (void *data, const void *host_ptr, void *device_ptr, size_t cb);
  void (*write_rect) (void *data, const void *host_ptr, void *device_ptr,
                      const size_t *buffer_origin,
                      const size_t *host_origin, 
                      const size_t *region,
                      size_t buffer_row_pitch,
                      size_t buffer_slice_pitch,
                      size_t host_row_pitch,
                      size_t host_slice_pitch);
  void (*copy) (void *data, const void *src_ptr, const void *dst_ptr, size_t cb);
  void (*copy_rect) (void *data, const void *src_ptr, void *dst_ptr,
                     const size_t *src_origin,
                     const size_t *dst_origin, 
                     const size_t *region,
                     size_t src_row_pitch,
                     size_t src_slice_pitch,
                     size_t dst_row_pitch,
                     size_t dst_slice_pitch);
  void (*run) (void *data, const char *bytecode,
	       cl_kernel kernel,
	       struct pocl_context *pc);
  void *data;
};

struct _cl_platform_id {
  int magic;
}; 

struct _cl_context {
  POCL_OBJECT;
  /* queries */
  cl_device_id *devices;
  const cl_context_properties *properties;
  /* implementation */
  unsigned num_devices;
};

struct _cl_command_queue {
  POCL_OBJECT;
  /* queries */
  cl_context context;
  cl_device_id device;
  cl_command_queue_properties properties;
  /* implementation */
};

typedef struct _cl_mem cl_mem_t;

struct _cl_mem {
  POCL_OBJECT;
  /* queries */
  cl_mem_object_type type;
  cl_mem_flags flags;
  size_t size;
  void *mem_host_ptr;
  cl_uint map_count;
  cl_context context;
  /* implementation */
  void **device_ptrs;
  /* in case this is a sub buffer, this points to the parent
     buffer */
  cl_mem_t *parent;
};

struct _cl_program {
  POCL_OBJECT;
  /* queries */
  cl_context context;
  cl_uint num_devices;
  cl_device_id *devices;
  char *source;
  size_t binary_size; /* same binary for all devices.  */
  unsigned char *binary; /* same binary for all devices.  */
  /* implementation */
  cl_kernel kernels;
};

struct _cl_kernel {
  POCL_OBJECT;
  /* queries */
  const char *function_name;
  cl_uint num_args;
  cl_context context;
  cl_program program;
  /* implementation */
  lt_dlhandle dlhandle;
  cl_int *arg_is_pointer;
  cl_int *arg_is_local;
  cl_uint num_locals;
  struct pocl_argument *arguments;
  struct _cl_kernel *next;
};

#endif /* POCL_CL_H */
