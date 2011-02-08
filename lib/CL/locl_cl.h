/* locl_cl.h - local runtime library declarations.

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

#ifndef LOCL_CL_H
#define LOCL_CL_H

#include "config.h"
#include <stdio.h>
#include <ltdl.h>
#include "CL/opencl.h"

#define LOCL_FILENAME_LENGTH 128

#define LOCL_BUILD "locl-build"
#define LOCL_KERNEL "locl-kernel"
#define LOCL_WORKGROUP "locl-workgroup"

#define LOCL_ERROR(x) if (errcode_ret != NULL) {*errcode_ret = (x); return NULL;}

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
  void *(*malloc)(void *data, cl_mem_flags flags,
		 size_t size, void *host_ptr);
  void (*free)(void *data, void *ptr);
  void (*read)(void *data, void *host_ptr, void *device_ptr);
  void (*run)(void *data, const char *bytecode,
	      cl_kernel kernel,
	      size_t x, size_t y, size_t z);
  void *data;
};

struct _cl_context {
  /* queries */
  cl_uint reference_count;
  cl_device_id *devices;
  const cl_context_properties *properties;
  /* implementation */
  unsigned num_devices;
};

struct _cl_command_queue {
  /* queries */
  cl_context context;
  cl_device_id device;
  cl_uint reference_count;
  cl_command_queue_properties properties;
  /* implementation */
};

struct _cl_mem {
  /* queries */
  cl_mem_object_type type;
  cl_mem_flags flags;
  size_t size;
  void *mem_host_ptr;
  cl_uint map_count;
  cl_uint reference_count;
  cl_context context;
  /* implementation */
  void **device_ptrs;
};

struct _cl_program {
  /* queries */
  cl_uint reference_count;
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
  /* queries */
  const char *function_name;
  cl_uint num_args;
  cl_uint reference_count;
  cl_context context;
  cl_program program;
  /* implementation */
  const char trampoline_filename[LOCL_FILENAME_LENGTH];
  lt_dlhandle dlhandle;
  struct _cl_kernel *next;
};

struct locl_argument {
  int is_local;
  size_t size;
  void *value;
};

typedef void (*workgroup) (size_t, size_t, size_t);

#endif /* LOCL_CL_H */
