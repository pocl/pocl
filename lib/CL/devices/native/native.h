/* native.h - native device declarations.

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

#ifndef POCL_NATIVE_H
#define POCL_NATIVE_H

#include "pocl_cl.h"

void pocl_native_init (cl_device_id device);
void *pocl_native_malloc (void *data, cl_mem_flags flags,
			  size_t size, void *host_ptr);
void pocl_native_free (void *data, void *ptr);
void pocl_native_read (void *data, void *host_ptr, void *device_ptr, size_t cb);
void pocl_native_run (void *data, const char *bytecode,
		      cl_kernel kernel,
		      size_t x, size_t y, size_t z);

extern size_t pocl_native_max_work_item_sizes[];

#define POCL_DEVICES_NATIVE {						\
  CL_DEVICE_TYPE_GPU, /* type */					\
  0, /* vendor_id */							\
  0, /* max_compute_units */						\
  1, /* max_work_item_dimensions */					\
  pocl_native_max_work_item_sizes, /* max_work_item_sizes */		\
  1, /*max_work_group_size */						\
  0, /* preferred_vector_width_char */					\
  0, /* preferred_vector_width_shortr */				\
  0, /* preferred_vector_width_int */					\
  0, /* preferred_vector_width_long */					\
  0, /* preferred_vector_width_float */					\
  0, /* preferred_vector_width_double */				\
  0, /* max_clock_frequency */						\
  0, /* address_bits */							\
  0, /* max_mem_alloc_size */						\
  CL_FALSE, /* image_support */						\
  0, /* max_read_image_args */						\
  0, /* max_write_image_args */						\
  0, /*image2d_max_width */						\
  0, /* image2d_max_height */						\
  0, /* image3d_max_width */						\
  0, /* image3d_max_height */						\
  0, /* image3d_max_depth */						\
  0, /* max_samplers */							\
  0, /* max_parameter_size */						\
  0, /* mem_base_addr_align */						\
  0, /* min_data_type_align_size */					\
  CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, /* single_fp_config */	\
  CL_NONE, /* global_mem_cache_type */					\
  0, /* global_mem_cacheline_size */					\
  0, /* global_mem_cache_size */					\
  0, /* global_mem_size */						\
  0, /* max_constant_buffer_size */					\
  0, /* max_constant_args */						\
  CL_GLOBAL, /* local_mem_type */					\
  0, /* local_mem_size */						\
  CL_FALSE, /* error_correction_support */				\
  0, /* profiling_timer_resolution */					\
  CL_FALSE, /* endian_little */						\
  CL_TRUE, /* available */						\
  CL_TRUE, /* compiler_available */					\
  CL_EXEC_KERNEL, /*execution_capabilities */				\
  CL_QUEUE_PROFILING_ENABLE, /* queue_properties */			\
  0, /* platform */							\
  "native", /* name */							\
  "pocl", /* vendor */							\
  "0.1", /* driver_version */						\
  "FULL_PROFILE", /* profile */						\
  "OpenCL 1.0 pocl", /* version */					\
  "", /* extensions */							\
  /* implementation */							\
    pocl_native_init, /* init */                                        \
  pocl_native_malloc, /* malloc */					\
  pocl_native_free, /* free */						\
  pocl_native_read, /* read */						\
  pocl_native_run, /* run */						\
  NULL /* data */							\
}

#endif /* POCL_NATIVE_H */
