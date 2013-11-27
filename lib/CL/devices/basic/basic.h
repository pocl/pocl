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
  .type = CL_DEVICE_TYPE_CPU,					        \
  .vendor_id = 0, 						        \
  .max_compute_units = 1, 						\
  .max_work_item_dimensions = 3, 					\
  .max_work_item_sizes = {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX},          \
  .max_work_group_size = 1024, 				                \
  .preferred_wg_size_multiple = 8,                                      \
  .preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR  ,   \
  .preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT ,  \
  .preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT   ,    \
  .preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG  ,   \
  .preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT ,  \
  .preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE, \
  .preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF  ,   \
  /* TODO: figure out what the difference between preferred and native widths are. */ \
  .preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR  ,   \
  .preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT ,  \
  .preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT   ,    \
  .preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG  ,   \
  .preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT ,  \
  .preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE, \
  .preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF  ,   \
  .max_clock_frequency = 0, 						\
  .address_bits = POCL_DEVICE_ADDRESS_BITS, 			        \
  .max_mem_alloc_size = 0, 						\
  .image_support = CL_TRUE, 						\
  .max_read_image_args = 0, 						\
  .max_write_image_args = 0, 						\
  .image2d_max_width = 0, 						\
  .image2d_max_height = 0, 						\
  .image3d_max_width = 0, 						\
  .image3d_max_height = 0, 						\
  .image3d_max_depth = 0, 						\
  .image_max_buffer_size = 0, 					        \
  .image_max_array_size = 0, 						\
  .max_samplers = 0, 							\
  .max_parameter_size = 1024, 						\
  .mem_base_addr_align = 0, 						\
  .min_data_type_align_size = 0, 					\
  .half_fp_config = 0, 	                                                \
  .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, 	        \
  .double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, 	        \
  .global_mem_cache_type = CL_NONE, 					\
  .global_mem_cacheline_size = 0, 					\
  .global_mem_cache_size = 0, 					        \
  .global_mem_size = 0, 						\
  .max_constant_buffer_size = 0, 					\
  .max_constant_args = 0, 						\
  .local_mem_type = CL_GLOBAL, 					        \
  .local_mem_size = 0, 						        \
  .error_correction_support = CL_FALSE, 				\
  .host_unified_memory = CL_TRUE,                                       \
  .profiling_timer_resolution = 0,					\
  .endian_little = !(WORDS_BIGENDIAN), 				        \
  .available = CL_TRUE, 						\
  .compiler_available = CL_TRUE, 					\
  .execution_capabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL, 	\
  .queue_properties = CL_QUEUE_PROFILING_ENABLE, 			\
  .platform = 0, 							\
  .device_partition_properties = {0},                                   \
  .printf_buffer_size = 0,						\
  .short_name = "basic", 						\
  .long_name = 0, 							\
  .vendor = "pocl",							\
  .profile = "FULL_PROFILE", 						\
  .extensions = "",							\
  /* implementation */							\
  .uninit = pocl_basic_uninit,                                          \
  .init = pocl_basic_init,                                              \
  .malloc = pocl_basic_malloc,					        \
  .free = pocl_basic_free,						\
  .read = pocl_basic_read,						\
  .read_rect = pocl_basic_read_rect,				        \
  .write = pocl_basic_write,					        \
  .write_rect = pocl_basic_write_rect,				        \
  .copy = pocl_basic_copy,						\
  .copy_rect = pocl_basic_copy_rect,				        \
  .fill_rect = pocl_basic_fill_rect,                                    \
  .map_mem = pocl_basic_map_mem,                                        \
  .run = pocl_basic_run,                                                \
  .run_native = pocl_basic_run_native,					\
  .get_timer_value = pocl_basic_get_timer_value,                        \
  .get_supported_image_formats = pocl_basic_get_supported_image_formats,\
  .llvm_target_triplet = OCL_KERNEL_TARGET,                             \
  .llvm_cpu = OCL_KERNEL_TARGET_CPU,                                    \
  .has_64bit_long = 1									\
}

#endif /* POCL_BASIC_H */
