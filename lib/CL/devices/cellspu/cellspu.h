/* cellspu.h - a pocl device driver for Cell SPU.

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

#ifndef POCL_CELLSPU_H
#define POCL_CELLSPU_H

#include "pocl_cl.h"
#include "pocl_icd.h"
#include "bufalloc.h"

#include "prototypes.inc"

/* simplistic linker script: 
 * this is the SPU local address where 'OpenCL global' memory starts.
 * (if we merge the spus to a single device, this is the 'OpenCL local' memory
 * 
 * The idea is to allocate
 * 64k (0-64k) for text.
 * 128k (64k-192k) for Opencl local memory.
 * 64k (192k-256k) for stack + heap (if any)
 * 
 * I was unable to place the stack to start at 0x20000, thus the "unclean" division.
 */
#define CELLSPU_OCL_BUFFERS_START 0x10000
#define CELLSPU_OCL_BUFFERS_SIZE  0x20000
#define CELLSPU_KERNEL_CMD_ADDR   0x30000
//#define CELLSPU_OCL_KERNEL_ADDRESS 0x2000


#ifdef __cplusplus
extern "C" {
#endif

GEN_PROTOTYPES (cellspu)

#define POCL_DEVICES_CELLSPU {						\
  POCL_DEVICE_ICD_DISPATCH     						\
  POCL_OBJECT_INIT, \
  CL_DEVICE_TYPE_ACCELERATOR, /* type */					\
  0, /* vendor_id */							\
  1, /* max_compute_units */						\
  3, /* max_work_item_dimensions */					\
  {CL_INT_MAX, CL_INT_MAX, CL_INT_MAX}, /* max_work_item_sizes */		\
  1024, /* max_work_group_size */					\
  8, /* preferred_wg_size_multiple */                                \
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
  100, /* max_clock_frequency */						\
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
  CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN, /* double_fp_config */       \
  CL_NONE, /* global_mem_cache_type */					\
  0, /* global_mem_cacheline_size */					\
  0, /* global_mem_cache_size */					\
  0, /* global_mem_size */						\
  0, /* max_constant_buffer_size */					\
  0, /* max_constant_args */						\
  CL_GLOBAL, /* local_mem_type */					\
  0, /* local_mem_size */						\
  CL_FALSE, /* error_correction_support */				\
  CL_TRUE, /* host_unified_memory */                \
  0, /* profiling_timer_resolution */					\
  CL_FALSE, /* endian_little */						\
  CL_TRUE, /* available */						\
  CL_TRUE, /* compiler_available */					\
  CL_EXEC_KERNEL, /*execution_capabilities */				\
  CL_QUEUE_PROFILING_ENABLE, /* queue_properties */			\
  0, /* platform */							\
  "cellspu", /* name */							\
  "STI", /* vendor */							\
  PACKAGE_VERSION, /* driver_version */						\
  "EMBEDDED_PROFILE", /* profile */						\
  "OpenCL 1.2 pocl", /* version */					\
  "", /* extensions */							\
  /* implementation */							\
  pocl_cellspu_uninit, /* init */                                     \
  pocl_cellspu_init, /* init */                                       \
  pocl_cellspu_malloc, /* malloc */					\
  pocl_cellspu_create_sub_buffer, \
  pocl_cellspu_free, /* free */						\
  pocl_cellspu_read, /* read */						\
  pocl_cellspu_read_rect, /* read_rect */				\
  pocl_cellspu_write, /* write */					\
  pocl_cellspu_write_rect, /* write_rect */				\
  pocl_cellspu_copy, /* copy */						\
  pocl_cellspu_copy_rect, /* copy_rect */				\
  pocl_cellspu_map_mem,                               \
  NULL, /* unmap_mem is a NOP */                    \
  pocl_cellspu_run, /* run */                         \
  pocl_cellspu_get_timer_value,                \
  NULL, /*pocl_cellspu_build_program */	 \
  NULL, /* data */                               \
  "cellspu", /* kernel_lib_target (forced kernel library dir) */    \
  "cellspu-v0", /* llvm_target_triplet */               \
  0 /* dev_id */ \
}

#ifdef __cplusplus
}
#endif

#endif /* POCL_CELLSPU_H */
