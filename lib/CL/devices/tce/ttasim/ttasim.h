/* ttasim.h - a pocl device driver for simulating TTA devices using TCE's ttasim

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#ifndef POCL_TTASIM_H
#define POCL_TTASIM_H

#include "pocl_cl.h"
#include "pocl_icd.h"
#include "bufalloc.h"

#include "tce_common.h"
#include "prototypes.inc"

#ifdef __cplusplus
extern "C" {
#endif

GEN_PROTOTYPES (ttasim)

#define POCL_DEVICES_TTASIM {						\
  .type = CL_DEVICE_TYPE_GPU, 				                \
  .max_compute_units = 1, 			                        \
  .max_work_item_dimensions = 3, 		                        \
  .max_work_item_sizes = {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX},          \
  .max_work_group_size = 8192, 				                \
  .preferred_wg_size_multiple = 8,                                      \
  .preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR  ,    \
  .preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT ,   \
  .preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT   ,     \
  .preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG  ,    \
  .preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT ,   \
  .preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE,  \
  .preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF,      \
  /* TODO: figure out what the difference between preferred and native widths are. */ \
  .preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR  ,    \
  .preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT ,   \
  .preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT   ,     \
  .preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG  ,    \
  .preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT ,   \
  .preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE,  \
  .preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF  ,    \
  .max_clock_frequency = 100, 						\
  .image_support = CL_FALSE, 						\
  .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, 	        \
  .double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN,           \
  .global_mem_cache_type = CL_NONE, 					\
  .local_mem_type = CL_GLOBAL, 					        \
  .error_correction_support = CL_FALSE, 				\
  .host_unified_memory = CL_FALSE,                                      \
  .endian_little = CL_FALSE, 						\
  .available = CL_TRUE, 						\
  .compiler_available = CL_TRUE, 					\
  .execution_capabilities = CL_EXEC_KERNEL, 				\
  .queue_properties = CL_QUEUE_PROFILING_ENABLE, 			\
  .device_partition_properties = {0},                                   \
  .short_name = "ttasim", 					        \
  .vendor = "TTA-Based Co-design Environment", 			        \
  .profile = "EMBEDDED_PROFILE", 				        \
  .extensions = "", 							\
  /* implementation */							\
  .uninit = pocl_ttasim_uninit,                                         \
  .init = pocl_ttasim_init,                                             \
  .malloc = pocl_tce_malloc, 					        \
  .create_sub_buffer = pocl_tce_create_sub_buffer,                      \
  .free = pocl_tce_free, 						\
  .read = pocl_tce_read, 						\
  .read_rect = pocl_tce_read_rect, 				        \
  .write = pocl_tce_write, 					        \
  .write_rect = pocl_tce_write_rect, 				        \
  .copy = pocl_tce_copy, 						\
  .copy_rect = pocl_tce_copy_rect, 				        \
  .map_mem = pocl_tce_map_mem,                                          \
  .run = pocl_tce_run,                                                  \
  .get_timer_value = pocl_ttasim_get_timer_value,                       \
  .build_program = pocl_tce_build_program,                              \
  .build = pocl_tce_init_build,                                         \
  .llvm_target_triplet = "tce-tut-llvm",                                \
  .has_64bit_long = 1				  \
}

#ifdef __cplusplus
}
#endif

#endif /* POCL_TTASIM_H */
