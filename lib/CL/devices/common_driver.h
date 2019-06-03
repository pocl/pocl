#include "pocl_export.h"

#ifndef POCL_COMMON_DRIVER_H
#define POCL_COMMON_DRIVER_H

#ifdef __cplusplus
extern "C"
{
#endif

POCL_EXPORT
  void pocl_driver_read (void *data, void *__restrict__ dst_host_ptr,
                         pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                         size_t offset, size_t size);
POCL_EXPORT
  void pocl_driver_read_rect (void *data, void *__restrict__ dst_host_ptr,
                              pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                              const size_t *buffer_origin,
                              const size_t *host_origin, const size_t *region,
                              size_t buffer_row_pitch,
                              size_t buffer_slice_pitch, size_t host_row_pitch,
                              size_t host_slice_pitch);
POCL_EXPORT
  void pocl_driver_write (void *data, const void *__restrict__ src_host_ptr,
                          pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                          size_t offset, size_t size);
POCL_EXPORT
  void pocl_driver_write_rect (void *data,
                               const void *__restrict__ src_host_ptr,
                               pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                               const size_t *buffer_origin,
                               const size_t *host_origin, const size_t *region,
                               size_t buffer_row_pitch,
                               size_t buffer_slice_pitch,
                               size_t host_row_pitch, size_t host_slice_pitch);
POCL_EXPORT
  void pocl_driver_copy (void *data, pocl_mem_identifier *dst_mem_id,
                         cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                         cl_mem src_buf, size_t dst_offset, size_t src_offset,
                         size_t size);
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
  int pocl_driver_build_source (cl_program program, cl_uint device_i,
                                cl_uint num_devices,
                                const cl_device_id *device_list,
                                cl_uint num_input_headers,
                                const cl_program *input_headers,
                                const char **header_include_names,
                                int link_program);
POCL_EXPORT
  int pocl_driver_build_binary (cl_program program, cl_uint device_i,
                                cl_uint num_devices,
                                const cl_device_id *device_list,
                                int link_program, int spir_build);
POCL_EXPORT
  int pocl_driver_link_program (cl_program program, cl_uint device_i,
                                cl_uint num_input_programs,
                                const cl_program *input_programs,
                                int create_library);
POCL_EXPORT
  int pocl_driver_free_program (cl_device_id device, cl_program program,
                                unsigned program_device_i);
POCL_EXPORT
  int pocl_driver_setup_metadata (cl_device_id device, cl_program program,
                                  unsigned program_device_i);
POCL_EXPORT
  int pocl_driver_supports_binary (cl_device_id device, const size_t length,
                                   const char *binary);
POCL_EXPORT
  int pocl_driver_build_poclbinary (cl_program program, cl_uint device_i);

#ifdef __cplusplus
}
#endif

#endif
