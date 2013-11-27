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
#include "pocl_icd.h"
#include "config.h"

#ifndef WORDS_BIGENDIAN
#define WORDS_BIGENDIAN 0
#endif

#include "prototypes.inc"
GEN_PROTOTYPES (pthread)

#include "prototypes.inc"
GEN_PROTOTYPES (basic)

#define POCL_DEVICES_PTHREAD {	 					\
  .type = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_DEFAULT, 			\
  .max_work_item_dimensions = 3, 					\
  /* This could be SIZE_T_MAX, but setting it to INT_MAX should suffice, */ \
  /* and may avoid errors in user code that uses int instead of size_t */ \
  .max_work_item_sizes = {1024, 1024, 1024},                            \
  .max_work_group_size = 1024, 					        \
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
  .address_bits = POCL_DEVICE_ADDRESS_BITS, 				\
  .image_support = CL_TRUE, 						\
  .max_parameter_size = 1024, 						\
  .single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, 	        \
  .double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, 	        \
  .global_mem_cache_type = CL_NONE, 					\
  .local_mem_type = CL_GLOBAL, 					        \
  .error_correction_support = CL_FALSE, 				\
  .host_unified_memory = CL_TRUE,                                       \
  .endian_little = !(WORDS_BIGENDIAN), 				        \
  .available = CL_TRUE, 						\
  .compiler_available = CL_TRUE, 					\
  .execution_capabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL, 	\
  .queue_properties = CL_QUEUE_PROFILING_ENABLE, 			\
  .short_name = "pthread", 						\
  .vendor = "pocl", 							\
  .profile = "FULL_PROFILE", 						\
  .extensions = "", 							\
  /* implementation */							\
  .uninit = pocl_pthread_uninit,                                        \
  .init = pocl_pthread_init,                                            \
  .malloc = pocl_pthread_malloc, 					\
  .free = pocl_pthread_free, 						\
  .read = pocl_pthread_read, 						\
  .read_rect = pocl_basic_read_rect, 				        \
  .write = pocl_pthread_write, 					        \
  .write_rect = pocl_basic_write_rect, 				        \
  .copy = pocl_pthread_copy, 						\
  .copy_rect = pocl_pthread_copy_rect, 				        \
  .fill_rect = pocl_basic_fill_rect,                                    \
  .map_mem = pocl_basic_map_mem,                                        \
  .run = pocl_pthread_run,                                              \
  .run_native = pocl_basic_run_native, 					\
  .get_timer_value = pocl_basic_get_timer_value,                        \
  .get_supported_image_formats = pocl_basic_get_supported_image_formats,\
  .llvm_target_triplet = OCL_KERNEL_TARGET,                             \
  .llvm_cpu = OCL_KERNEL_TARGET_CPU,                                    \
  .has_64bit_long = 1									\
}


#endif /* POCL_PTHREAD_H */
