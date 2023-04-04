/* OpenCL runtime library: shared functions

   Copyright (c) 2016 Michal Babej / Tampere University of Technology

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

#ifndef POCL_SHARED_H
#define POCL_SHARED_H

#include "config.h"

#include "pocl_cl.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

void pocl_check_uninit_devices ();

cl_program create_program_skeleton (cl_context context, cl_uint num_devices,
                                    const cl_device_id *device_list,
                                    const size_t *lengths,
                                    const unsigned char **binaries,
                                    cl_int *binary_status, cl_int *errcode_ret,
                                    int allow_empty_binaries);

cl_mem pocl_create_image_internal (cl_context context, cl_mem_flags flags,
                                   const cl_image_format *image_format,
                                   const cl_image_desc *image_desc,
                                   void *host_ptr, cl_int *errcode_ret,
                                   cl_GLenum gl_target, cl_GLint gl_miplevel,
                                   cl_GLuint gl_texture,
                                   CLeglDisplayKHR egl_display,
                                   CLeglImageKHR egl_image
);

cl_int
compile_and_link_program(int compile_program,
                         int link_program,
                         cl_program program,
                         cl_uint num_devices,
                         const cl_device_id *device_list,
                         const char *options,
                         cl_uint num_input_headers,
                         const cl_program *input_headers,
                         const char **header_include_names,
                         cl_uint num_input_programs,
                         const cl_program *input_programs,
                         void (CL_CALLBACK *pfn_notify) (cl_program program,
                                                         void *user_data),
                         void *user_data);

int context_set_properties (cl_context context,
                            const cl_context_properties *properties);

cl_mem pocl_create_memobject (cl_context context, cl_mem_flags flags,
                              size_t size, cl_mem_object_type type,
                              int *device_image_support, void *host_ptr,
                              int host_ptr_is_svm, cl_int *errcode_ret);

cl_int pocl_kernel_copy_args (cl_kernel kernel,
                              struct pocl_argument *src_arguments,
                              _cl_command_run *command);

cl_int pocl_ndrange_kernel_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    const cl_ndrange_kernel_command_properties_khr *properties,
    cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset,
    const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_items_in_wait_list, const cl_event *event_wait_list,
    cl_event *event_p, const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point_p, _cl_command_node **cmd);

cl_int pocl_rect_copy (cl_command_buffer_khr command_buffer,
                       cl_command_queue command_queue,
                       cl_command_type command_type,
                       cl_mem src,
                       cl_int src_is_image,
                       cl_mem dst,
                       cl_int dst_is_image,
                       const size_t *src_origin,
                       const size_t *dst_origin,
                       const size_t *region,
                       size_t *src_row_pitch,
                       size_t *src_slice_pitch,
                       size_t *dst_row_pitch,
                       size_t *dst_slice_pitch,
                       cl_uint num_items_in_wait_list,
                       const cl_event *event_wait_list,
                       cl_event *event,
                       const cl_sync_point_khr *sync_point_wait_list,
                       cl_sync_point_khr *sync_point,
                       _cl_command_node **cmd);

cl_int pocl_copy_buffer_common (cl_command_buffer_khr command_buffer,
                                cl_command_queue command_queue,
                                cl_mem src_buffer,
                                cl_mem dst_buffer,
                                size_t src_offset,
                                size_t dst_offset,
                                size_t size,
                                cl_uint num_items_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event,
                                const cl_sync_point_khr *sync_point_wait_list,
                                cl_sync_point_khr *sync_point,
                                _cl_command_node **cmd);

cl_int pocl_copy_buffer_rect_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_buffer, cl_mem dst_buffer, const size_t *src_origin,
    const size_t *dst_origin, const size_t *region, size_t src_row_pitch,
    size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch,
    cl_uint num_items_in_wait_list, const cl_event *event_wait_list,
    cl_event *event, const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, _cl_command_node **cmd);

cl_int pocl_copy_buffer_to_image_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_buffer, cl_mem dst_image, size_t src_offset,
    const size_t *dst_origin, const size_t *region,
    cl_uint num_items_in_wait_list, const cl_event *event_wait_list,
    cl_event *event, const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, cl_mutable_command_khr *mutable_handle,
    _cl_command_node **cmd);

cl_int pocl_copy_image_to_buffer_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_image, cl_mem dst_buffer, const size_t *src_origin,
    const size_t *region, size_t dst_offset, cl_uint num_items_in_wait_list,
    const cl_event *event_wait_list, cl_event *event,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, cl_mutable_command_khr *mutable_handle,
    _cl_command_node **cmd);

cl_int pocl_copy_image_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_image, cl_mem dst_image, const size_t *src_origin,
    const size_t *dst_origin, const size_t *region,
    cl_uint num_items_in_wait_list, const cl_event *event_wait_list,
    cl_event *event, const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, _cl_command_node **cmd);

cl_int pocl_fill_buffer_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem buffer, const void *pattern, size_t pattern_size, size_t offset,
    size_t size, cl_uint num_items_in_wait_list,
    const cl_event *event_wait_list, cl_event *event,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, _cl_command_node **cmd);

cl_int pocl_fill_image_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem image, const void *fill_color, const size_t *origin,
    const size_t *region, cl_uint num_items_in_wait_list,
    const cl_event *event_wait_list, cl_event *event,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, cl_mutable_command_khr *mutable_handle,
    _cl_command_node **cmd);

cl_int pocl_svm_memcpy_common (cl_command_buffer_khr command_buffer,
                               cl_command_queue command_queue,
                               cl_command_type command_type,
                               void *dst_ptr, const void *src_ptr, size_t size,
                               cl_uint num_items_in_wait_list,
                               const cl_event *event_wait_list, cl_event *event,
                               const cl_sync_point_khr *sync_point_wait_list,
                               cl_sync_point_khr *sync_point,
                               _cl_command_node **cmd);


cl_int pocl_svm_memfill_common (cl_command_buffer_khr command_buffer,
                                cl_command_queue command_queue,
                                cl_command_type command_type,
                                void *svm_ptr, size_t size, const void *pattern,
                                size_t pattern_size,
                                cl_uint num_items_in_wait_list,
                                const cl_event *event_wait_list, cl_event *event,
                                const cl_sync_point_khr *sync_point_wait_list,
                                cl_sync_point_khr *sync_point,
                                _cl_command_node **cmd);

cl_int
pocl_svm_memcpy_rect_common (cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             void *dst_ptr,
                             const void *src_ptr,
                             const size_t *src_origin,
                             const size_t *dst_origin,
                             const size_t *region,
                             size_t src_row_pitch,
                             size_t src_slice_pitch,
                             size_t dst_row_pitch,
                             size_t dst_slice_pitch,
                             cl_uint num_items_in_wait_list,
                             const cl_event *event_wait_list,
                             cl_event *event,
                             const cl_sync_point_khr *sync_point_wait_list,
                             cl_sync_point_khr *sync_point,
                             _cl_command_node **cmd);

cl_int
pocl_svm_memfill_rect_common (cl_command_buffer_khr command_buffer,
                              cl_command_queue command_queue,
                              void *svm_ptr,
                              const size_t *origin,
                              const size_t *region,
                              size_t row_pitch,
                              size_t slice_pitch,
                              const void *pattern,
                              size_t pattern_size,
                              cl_uint num_items_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event,
                              const cl_sync_point_khr *sync_point_wait_list,
                              cl_sync_point_khr *sync_point,
                              _cl_command_node **cmd);

cl_int pocl_read_buffer_common (cl_command_buffer_khr command_buffer,
                                cl_command_queue command_queue,
                                cl_mem buffer,
                                size_t offset,
                                size_t size,
                                void *ptr,
                                cl_uint num_items_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event,
                                const cl_sync_point_khr *sync_point_wait_list,
                                cl_sync_point_khr *sync_point,
                                _cl_command_node **cmd);

cl_int pocl_write_buffer_common (cl_command_buffer_khr command_buffer,
                                 cl_command_queue command_queue,
                                 cl_mem buffer,
                                 size_t offset,
                                 size_t size,
                                 const void *ptr,
                                 cl_uint num_items_in_wait_list,
                                 const cl_event *event_wait_list,
                                 cl_event *event,
                                 const cl_sync_point_khr *sync_point_wait_list,
                                 cl_sync_point_khr *sync_point,
                                 _cl_command_node **cmd);

cl_int pocl_read_image_common (cl_command_buffer_khr command_buffer,
                               cl_command_queue command_queue,
                               cl_mem image,
                               const size_t *origin, /* [3] */
                               const size_t *region, /* [3] */
                               size_t row_pitch,
                               size_t slice_pitch,
                               void *ptr,
                               cl_uint num_items_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event,
                               const cl_sync_point_khr *sync_point_wait_list,
                               cl_sync_point_khr *sync_point,
                               _cl_command_node **cmd);

cl_int pocl_write_image_common (cl_command_buffer_khr command_buffer,
                                cl_command_queue command_queue,
                                cl_mem image,
                                const size_t *origin, /* [3] */
                                const size_t *region, /* [3] */
                                size_t row_pitch,
                                size_t slice_pitch,
                                const void *ptr,
                                cl_uint num_items_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event,
                                const cl_sync_point_khr *sync_point_wait_list,
                                cl_sync_point_khr *sync_point,
                                _cl_command_node **cmd);

cl_int
pocl_read_buffer_rect_common (cl_command_buffer_khr command_buffer,
                              cl_command_queue command_queue,
                              cl_mem buffer,
                              const size_t *buffer_origin,
                              const size_t *host_origin,
                              const size_t *region,
                              size_t buffer_row_pitch,
                              size_t buffer_slice_pitch,
                              size_t host_row_pitch,
                              size_t host_slice_pitch,
                              void *ptr,
                              cl_uint num_items_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event,
                              const cl_sync_point_khr *sync_point_wait_list,
                              cl_sync_point_khr *sync_point,
                              _cl_command_node **cmd);

cl_int
pocl_write_buffer_rect_common (cl_command_buffer_khr command_buffer,
                               cl_command_queue command_queue,
                               cl_mem buffer,
                               const size_t *buffer_origin,
                               const size_t *host_origin,
                               const size_t *region,
                               size_t buffer_row_pitch,
                               size_t buffer_slice_pitch,
                               size_t host_row_pitch,
                               size_t host_slice_pitch,
                               const void *ptr,
                               cl_uint num_items_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event,
                               const cl_sync_point_khr *sync_point_wait_list,
                               cl_sync_point_khr *sync_point,
                               _cl_command_node **cmd);

/* this one is NOT implemented for command buffers */
cl_int pocl_svm_migrate_mem_common (cl_command_type command_type,
                                    cl_command_queue command_queue,
                                    cl_uint num_svm_pointers,
                                    const void **svm_pointers,
                                    const size_t *sizes,
                                    cl_mem_migration_flags flags,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event);

#define POCL_VALIDATE_WAIT_LIST_PARAMS                                        \
  do                                                                          \
    {                                                                         \
      if (command_buffer == NULL)                                             \
        {                                                                     \
          assert (sync_point_wait_list == NULL);                              \
          POCL_RETURN_ERROR_COND (                                            \
              (event_wait_list == NULL && num_items_in_wait_list > 0),        \
              CL_INVALID_EVENT_WAIT_LIST);                                    \
          POCL_RETURN_ERROR_COND (                                            \
              (event_wait_list != NULL && num_items_in_wait_list == 0),       \
              CL_INVALID_EVENT_WAIT_LIST);                                    \
        }                                                                     \
      else                                                                    \
        {                                                                     \
          assert (event_wait_list == NULL && event == NULL);                  \
        }                                                                     \
    }                                                                         \
  while (0)
/* sync point wait list is validated in pocl_create_recorded_command */

int
pocl_set_kernel_arg_pointer(cl_kernel kernel,
                            cl_uint arg_index,
                            const void *arg_value);

#define POCL_VALIDATE_WAIT_LIST_PARAMS                                        \
  do                                                                          \
    {                                                                         \
      if (command_buffer == NULL)                                             \
        {                                                                     \
          assert (sync_point_wait_list == NULL);                              \
          POCL_RETURN_ERROR_COND (                                            \
              (event_wait_list == NULL && num_items_in_wait_list > 0),        \
              CL_INVALID_EVENT_WAIT_LIST);                                    \
          POCL_RETURN_ERROR_COND (                                            \
              (event_wait_list != NULL && num_items_in_wait_list == 0),       \
              CL_INVALID_EVENT_WAIT_LIST);                                    \
        }                                                                     \
      else                                                                    \
        {                                                                     \
          assert (event_wait_list == NULL && event == NULL);                  \
        }                                                                     \
    }                                                                         \
  while (0)
/* sync point wait list is validated in pocl_create_recorded_command */


#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif // POCL_SHARED_H
