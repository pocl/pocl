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

void pocl_pthread_uninit (cl_device_id device);
void pocl_pthread_init (cl_device_id device);
void *pocl_pthread_malloc (void *data, cl_mem_flags flags,
			  size_t size, void *host_ptr);
void pocl_pthread_free (void *data, cl_mem_flags flags, void *ptr);
void pocl_pthread_read (void *data, void *host_ptr, const void *device_ptr, size_t cb);
void pocl_pthread_read_rect (void *data, void *host_ptr, void *device_ptr,
                             const size_t *buffer_origin,
                             const size_t *host_origin, 
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch);
void pocl_pthread_write (void *data, const void *host_ptr, void *device_ptr, size_t cb);
void pocl_pthread_write_rect (void *data, const void *host_ptr, void *device_ptr,
                              const size_t *buffer_origin,
                              const size_t *host_origin, 
                              const size_t *region,
                              size_t buffer_row_pitch,
                              size_t buffer_slice_pitch,
                              size_t host_row_pitch,
                              size_t host_slice_pitch);
void pocl_pthread_copy (void *data, const void *src_ptr, void *__restrict__ dst_ptr, size_t cb);
void pocl_pthread_copy_rect (void *data, const void *src_ptr, void *dst_ptr,
                             const size_t *src_origin,
                             const size_t *dst_origin, 
                             const size_t *region,
                             size_t src_row_pitch,
                             size_t src_slice_pitch,
                             size_t dst_row_pitch,
                             size_t dst_slice_pitch);
void pocl_pthread_run (void *data, const char *bytecode,
		      cl_kernel kernel,
		      struct pocl_context *pc);

void* 
pocl_pthread_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size);


extern size_t pocl_pthread_max_work_item_sizes[];

/* Determine preferred vector sizes */
#if defined(__AVX__)
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_CHAR   16
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_SHORT   8
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_INT     4
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_LONG    2
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_FLOAT   4
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_DOUBLE  2
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_CHAR      16
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_SHORT      8
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_INT        4
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_LONG       2
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_FLOAT      8
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_DOUBLE     4
#elif defined(__SSE2__)
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_CHAR   16
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_SHORT   8
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_INT     4
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_LONG    2
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_FLOAT   4
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_DOUBLE  2
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_CHAR      16
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_SHORT      8
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_INT        4
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_LONG       2
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_FLOAT      4
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_DOUBLE     2
#else
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_CHAR    1
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_SHORT   1
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_INT     1
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_LONG    1
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_FLOAT   1
#  define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_DOUBLE  1
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_CHAR       1
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_SHORT      1
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_INT        1
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_LONG       1
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_FLOAT      1
#  define POCL_DEVICES_PTHREAD_NATIVE_VECTOR_WIDTH_DOUBLE     1
#endif
/* Half is internally represented as short */
#define POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_HALF POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_SHORT

#define POCL_DEVICES_PTHREAD {						\
  CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU , /* type */					\
  0, /* vendor_id */							\
  0, /* max_compute_units */						\
  3, /* max_work_item_dimensions */					\
  pocl_pthread_max_work_item_sizes, /* max_work_item_sizes */		\
  CL_INT_MAX, /* max_work_group_size */					\
  POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_CHAR  , /* preferred_vector_width_char */ \
  POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_SHORT , /* preferred_vector_width_short */ \
  POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_INT   , /* preferred_vector_width_int */ \
  POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_LONG  , /* preferred_vector_width_long */ \
  POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_FLOAT , /* preferred_vector_width_float */ \
  POCL_DEVICES_PTHREAD_PREFERRED_VECTOR_WIDTH_DOUBLE, /* preferred_vector_width_double */ \
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
  CL_FALSE, /* endian_little */						\
  CL_TRUE, /* available */						\
  CL_TRUE, /* compiler_available */					\
  CL_EXEC_KERNEL, /*execution_capabilities */				\
  CL_QUEUE_PROFILING_ENABLE, /* queue_properties */			\
  0, /* platform */							\
  "pthread", /* name */							\
  "pocl", /* vendor */							\
  "0.1", /* driver_version */						\
  "FULL_PROFILE", /* profile */						\
  "OpenCL 1.0 pocl", /* version */					\
  "", /* extensions */							\
  /* implementation */							\
  pocl_pthread_uninit, /* init */                                     \
  pocl_pthread_init, /* init */                                       \
  pocl_pthread_malloc, /* malloc */					\
  pocl_pthread_free, /* free */						\
  pocl_pthread_read, /* read */						\
  pocl_pthread_read_rect, /* read_rect */				\
  pocl_pthread_write, /* write */					\
  pocl_pthread_write_rect, /* write_rect */				\
  pocl_pthread_copy, /* copy */						\
  pocl_pthread_copy_rect, /* copy_rect */				\
  pocl_pthread_map_mem,                               \
  NULL, /* unmap_mem is a NOP */                    \
  pocl_pthread_run, /* run */                         \
  NULL /* data */							\
}

#endif /* POCL_PTHREAD_H */
