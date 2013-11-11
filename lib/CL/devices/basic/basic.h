/* basic.h - a minimalistic pocl device driver layer implementation

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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
 * @file basic.h
 *
 * The purpose of the 'basic' device driver is to serve as an example of
 * a minimalistic (but still working) device driver for pocl.
 *
 * It is a "native device" without multithreading and uses the malloc
 * directly for buffer allocation etc.
 */

#ifndef POCL_BASIC_H
#define POCL_BASIC_H

#include "pocl_cl.h"
#include "pocl_icd.h"
#include "config.h"

#ifndef WORDS_BIGENDIAN
#define WORDS_BIGENDIAN 0
#endif

#include "prototypes.inc"
GEN_PROTOTYPES (basic)

#define POCL_DEVICES_BASIC {						\
  POCL_DEVICE_ICD_DISPATCH     						\
  POCL_OBJECT_INIT, \
  CL_DEVICE_TYPE_CPU, /* type */					\
  0, /* vendor_id */							\
  1, /* max_compute_units */						\
  3, /* max_work_item_dimensions */					\
  {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX}, /* max_work_item_sizes */       \
  1024, /* max_work_group_size */					\
  8, /* preferred_wg_size_multiple */                                   \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR  , /* preferred_vector_width_char */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT , /* preferred_vector_width_short */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT   , /* preferred_vector_width_int */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG  , /* preferred_vector_width_long */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT , /* preferred_vector_width_float */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE, /* preferred_vector_width_double */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF  , /* preferred_vector_width_half */ \
  /* TODO: figure out what the difference between preferred and native widths are. */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR  , /* preferred_vector_width_char */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT , /* preferred_vector_width_short */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT   , /* preferred_vector_width_int */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG  , /* preferred_vector_width_long */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT , /* preferred_vector_width_float */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE, /* preferred_vector_width_double */ \
  POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF  , /* preferred_vector_width_half */ \
  0, /* max_clock_frequency */						\
  POCL_DEVICE_ADDRESS_BITS, /* address_bits */							\
  0, /* max_mem_alloc_size */						\
  CL_TRUE, /* image_support */						\
  0, /* max_read_image_args */						\
  0, /* max_write_image_args */						\
  0, /* image2d_max_width */						\
  0, /* image2d_max_height */						\
  0, /* image3d_max_width */						\
  0, /* image3d_max_height */						\
  0, /* image3d_max_depth */						\
  0, /* image_max_buffer_size */					\
  0, /* image_max_array_size */						\
  0, /* max_samplers */							\
  1024, /* max_parameter_size */						\
  0, /* mem_base_addr_align */						\
  0, /* min_data_type_align_size */					\
  0, /* half_fp_config */	\
  CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, /* single_fp_config */	\
  CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, /* double_fp_config */	\
  CL_NONE, /* global_mem_cache_type */					\
  0, /* global_mem_cacheline_size */					\
  0, /* global_mem_cache_size */					\
  0, /* global_mem_size */						\
  0, /* max_constant_buffer_size */					\
  0, /* max_constant_args */						\
  CL_GLOBAL, /* local_mem_type */					\
  0, /* local_mem_size */						\
  CL_FALSE, /* error_correction_support */				\
  CL_TRUE, /* host_unified_memory */                  \
  0, /* profiling_timer_resolution */					\
  !(WORDS_BIGENDIAN), /* endian_little */				\
  CL_TRUE, /* available */						\
  CL_TRUE, /* compiler_available */					\
  CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL, /*execution_capabilities */				\
  CL_QUEUE_PROFILING_ENABLE, /* queue_properties */			\
  0, /* platform */							\
  {0}, /* device_partition_properties */ \
  0, /* printf_buffer_size */						\
  "basic", /* short_name */							\
  0, /* long_name */							\
  "pocl", /* vendor */							\
  PACKAGE_VERSION, /* driver_version */						\
  "FULL_PROFILE", /* profile */						\
  "OpenCL 1.2 pocl", /* version */					\
  "", /* extensions */							\
  /* implementation */							\
  pocl_basic_uninit, /* init */                                     \
  pocl_basic_init, /* init */                                       \
  pocl_basic_malloc, /* malloc */					\
  NULL, /* create_sub_buffer */ \
  pocl_basic_free, /* free */						\
  pocl_basic_read, /* read */						\
  pocl_basic_read_rect, /* read_rect */				\
  pocl_basic_write, /* write */					\
  pocl_basic_write_rect, /* write_rect */				\
  pocl_basic_copy, /* copy */						\
  pocl_basic_copy_rect, /* copy_rect */				\
  pocl_basic_fill_rect, /* fill_rect*/              \
  pocl_basic_map_mem,                                 \
  NULL, /* unmap_mem is a NOP */                    \
  pocl_basic_run, /* run */                         \
  pocl_basic_run_native, /* run_native */						\
  pocl_basic_get_timer_value,  /* get_timer_value */    \
  NULL, /* build_program */ \
  pocl_basic_get_supported_image_formats, /* get_supported_image_formats */ \
  NULL, /* data */                                      \
  OCL_KERNEL_TARGET, /* llvm_target_triplet */                           \
  0     /* dev_id */           \
}

#endif /* POCL_BASIC_H */
