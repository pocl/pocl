#include "pocl_export.h"

#ifndef POCL_COMMON_DRIVER_H
#define POCL_COMMON_DRIVER_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef void (*gvar_init_callback_t)(cl_program program, cl_uint dev_i,
                                     _cl_command_node *fake_cmd);

POCL_EXPORT
void pocl_driver_read (void *data, void *__restrict__ dst_host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                       size_t offset, size_t size);

/**
 * Helper function for implementing rect read with an host-side memcpy.
 */
POCL_EXPORT
void
pocl_driver_read_rect_memcpy (void *data,
                              void *__restrict__ const host_ptr,
                              char *device_ptr,
                              const size_t *__restrict__ const buffer_origin,
                              const size_t *__restrict__ const host_origin,
                              const size_t *__restrict__ const region,
                              size_t const buffer_row_pitch,
                              size_t const buffer_slice_pitch,
                              size_t const host_row_pitch,
                              size_t const host_slice_pitch);

/**
 * Driver hook-compatibile function for implementing rect read with
 * a host-side memcpy.
 *
 * Can be typically used directly for CPU (host) drivers.
 */
POCL_EXPORT
void pocl_driver_read_rect (void *data,
                            void *__restrict__ dst_host_ptr,
                            pocl_mem_identifier *src_mem_id,
                            cl_mem src_buf,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch);

POCL_EXPORT
void pocl_driver_write (void *data,
                        const void *__restrict__ src_host_ptr,
                        pocl_mem_identifier *dst_mem_id,
                        cl_mem dst_buf,
                        size_t offset,
                        size_t size);

/**
 * Helper function for implementing rect write with an host-side memcpy.
 */
POCL_EXPORT
void pocl_driver_write_rect_memcpy (
  void *dev_data,
  const void *__restrict__ const host_ptr,
  char *device_ptr,
  const size_t *__restrict__ const buffer_origin,
  const size_t *__restrict__ const host_origin,
  const size_t *__restrict__ const region,
  size_t const buffer_row_pitch,
  size_t const buffer_slice_pitch,
  size_t const host_row_pitch,
  size_t const host_slice_pitch);

/**
 * Driver hook-compatibile function for implementing rect write with
 * an host-side memcpy.
 *
 * Can be typically used directly for CPU (host) drivers.
 */
POCL_EXPORT
void pocl_driver_write_rect (void *data,
                             const void *__restrict__ src_host_ptr,
                             pocl_mem_identifier *dst_mem_id,
                             cl_mem dst_buf,
                             const size_t *buffer_origin,
                             const size_t *host_origin,
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch);
  POCL_EXPORT
  void pocl_driver_copy (void *data, pocl_mem_identifier *dst_mem_id,
                         cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                         cl_mem src_buf, size_t dst_offset, size_t src_offset,
                         size_t size);

POCL_EXPORT
  void pocl_driver_copy_with_size (void *data,
                                   pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                                   pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                                   pocl_mem_identifier *content_size_buf_mem_id,
                                   cl_mem content_size_buf,
                                   size_t dst_offset, size_t src_offset, size_t size);

POCL_EXPORT
  void pocl_driver_copy_rect (void *data, pocl_mem_identifier *dst_mem_id,
                              cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                              cl_mem src_buf, const size_t *dst_origin,
                              const size_t *src_origin, const size_t *region,
                              size_t dst_row_pitch, size_t dst_slice_pitch,
                              size_t src_row_pitch, size_t src_slice_pitch);
POCL_EXPORT
  void pocl_driver_memfill (void *data, pocl_mem_identifier *dst_mem_id,
                            cl_mem dst_buf, size_t size, size_t offset,
                            const void *__restrict__ pattern,
                            size_t pattern_size);
POCL_EXPORT
  cl_int pocl_driver_map_mem (void *data, pocl_mem_identifier *src_mem_id,
                              cl_mem src_buf, mem_mapping_t *map);
POCL_EXPORT
  cl_int pocl_driver_unmap_mem (void *data, pocl_mem_identifier *dst_mem_id,
                                cl_mem dst_buf, mem_mapping_t *map);

  POCL_EXPORT
  cl_int pocl_driver_get_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                                      cl_mem mem, mem_mapping_t *map);

  POCL_EXPORT
  cl_int pocl_driver_free_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                                       cl_mem mem, mem_mapping_t *map);

POCL_EXPORT
cl_int
pocl_driver_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr);

POCL_EXPORT
void pocl_driver_free (cl_device_id device, cl_mem mem);

POCL_EXPORT
void pocl_driver_svm_fill (cl_device_id dev,
                           void *__restrict__ svm_ptr,
                           size_t size,
                           void *__restrict__ pattern,
                           size_t pattern_size);

POCL_EXPORT
void pocl_driver_svm_copy (cl_device_id dev,
                           void *__restrict__ dst,
                           const void *__restrict__ src,
                           size_t size);

POCL_EXPORT
void pocl_driver_svm_fill_rect (cl_device_id dev,
                                void *__restrict__ svm_ptr,
                                const size_t *origin,
                                const size_t *region,
                                size_t row_pitch,
                                size_t slice_pitch,
                                void *__restrict__ pattern,
                                size_t pattern_size);

/**
 * Helper function for implementing write rect with an host-side memcpy.
 */
POCL_EXPORT
void pocl_driver_copy_rect_memcpy (cl_device_id dev,
                                   void *__restrict__ dst_ptr,
                                   const void *__restrict__ src_ptr,
                                   const size_t *__restrict__ const dst_origin,
                                   const size_t *__restrict__ const src_origin,
                                   const size_t *__restrict__ const region,
                                   size_t dst_row_pitch,
                                   size_t dst_slice_pitch,
                                   size_t src_row_pitch,
                                   size_t src_slice_pitch);

POCL_EXPORT
int pocl_driver_build_source (cl_program program,
                              cl_uint device_i,
                              cl_uint num_input_headers,
                              const cl_program *input_headers,
                              const char **header_include_names,
                              int link_program);
POCL_EXPORT
int pocl_driver_build_binary (cl_program program,
                              cl_uint device_i,
                              int link_program,
                              int spir_build);
POCL_EXPORT
int pocl_driver_link_program (cl_program program,
                              cl_uint device_i,
                              cl_uint num_input_programs,
                              const cl_program *input_programs,
                              int create_library);
POCL_EXPORT
int pocl_driver_free_program (cl_device_id device,
                              cl_program program,
                              unsigned program_device_i);
POCL_EXPORT
int pocl_driver_setup_metadata (cl_device_id device,
                                cl_program program,
                                unsigned program_device_i);
POCL_EXPORT
int pocl_driver_supports_binary (cl_device_id device,
                                 size_t length,
                                 const char *binary);
POCL_EXPORT
int pocl_driver_build_poclbinary (cl_program program, cl_uint device_i);

POCL_EXPORT
int pocl_driver_build_opencl_builtins (cl_program program, cl_uint device_i);

POCL_EXPORT
int pocl_driver_build_gvar_init_kernel (cl_program program,
                                        cl_uint dev_i,
                                        cl_device_id device,
                                        gvar_init_callback_t callback);

POCL_EXPORT
void pocl_cpu_gvar_init_callback (cl_program program,
                                  cl_uint dev_i,
                                  _cl_command_node *fake_cmd);
POCL_EXPORT
cl_int pocl_driver_get_synchronized_timestamps (cl_device_id dev,
                                                cl_ulong *dev_timestamp,
                                                cl_ulong *host_timestamp);


#ifdef __cplusplus
}
#endif

#endif
