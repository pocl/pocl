/* pthread.h - native pthreaded device declarations.

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

#ifndef POCL_PTHREAD_H
#define POCL_PTHREAD_H

#include "pocl_cl.h"

#include "prototypes.inc"
GEN_PROTOTYPES (pthread)

extern size_t pocl_pthread_max_work_item_sizes[];

#define POCL_DEVICES_PTHREAD {						\
  CL_DEVICE_TYPE_CPU, /* type */					\
  0, /* vendor_id */							\
  0, /* max_compute_units */						\
  3, /* max_work_item_dimensions */					\
  pocl_pthread_max_work_item_sizes, /* max_work_item_sizes */		\
  1024, /* max_work_group_size */					\
  8, /* preferred_wg_size_multiple */                                \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR  , /* preferred_vector_width_char */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT , /* preferred_vector_width_short */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT   , /* preferred_vector_width_int */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG  , /* preferred_vector_width_long */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT , /* preferred_vector_width_float */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE, /* preferred_vector_width_double */ \
  0, /* max_clock_frequency */						\
  0, /* address_bits */							\
  0, /* max_mem_alloc_size */						\
  CL_FALSE, /* image_support */						\
  0, /* max_read_image_args */						\
  0, /* max_write_image_args */						\
  0, /* image2d_max_width */						\
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
  CL_TRUE, /* endian_little: TODO: check from CPU id */						\
  CL_TRUE, /* available */						\
  CL_TRUE, /* compiler_available */					\
  CL_EXEC_KERNEL, /*execution_capabilities */				\
  CL_QUEUE_PROFILING_ENABLE, /* queue_properties */			\
  0, /* platform */							\
  "pthread", /* name */							\
  "pocl", /* vendor */							\
  PACKAGE_VERSION, /* driver_version */						\
  "FULL_PROFILE", /* profile */						\
  "OpenCL 1.2 pocl", /* version */					\
  "", /* extensions */							\
  /* implementation */							\
  pocl_pthread_uninit, /* uninit */                                     \
  pocl_pthread_init, /* init */                                       \
  pocl_pthread_malloc, /* malloc */					\
  NULL, /* create_sub_buffer */                   \
  pocl_pthread_free, /* free */						\
  pocl_pthread_read, /* read */						\
  pocl_pthread_read_rect, /* read_rect */				\
  pocl_pthread_write, /* write */					\
  pocl_pthread_write_rect, /* write_rect */				\
  pocl_pthread_copy, /* copy */						\
  pocl_pthread_copy_rect, /* copy_rect */				\
  pocl_basic_map_mem,                               \
  NULL, /* unmap_mem is a NOP */                    \
  pocl_pthread_run, /* run */                         \
  NULL, /* data */                                  \
  NULL,  /* kernel_lib_target (forced kernel library dir) */  \
  NULL, /* llvm_target_triplet */                         \
  0     /* dev_id */                                    \
}

#endif /* POCL_PTHREAD_H */
