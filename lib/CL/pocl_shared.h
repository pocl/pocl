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

cl_int pocl_rect_copy(cl_command_queue command_queue,
                      cl_command_type command_type,
                      cl_mem src,
                      cl_int src_is_image,
                      cl_mem dst,
                      cl_int dst_is_image,
                      const size_t *src_origin,
                      const size_t *dst_origin,
                      const size_t *region,
                      size_t src_row_pitch,
                      size_t src_slice_pitch,
                      size_t dst_row_pitch,
                      size_t dst_slice_pitch,
                      cl_uint num_events_in_wait_list,
                      const cl_event *event_wait_list,
                      cl_event *event,
                      _cl_command_node **cmd);

cl_int pocl_record_rect_copy (
    cl_command_queue command_queue, cl_command_type command_type, cl_mem src,
    cl_int src_is_image, cl_mem dst, cl_int dst_is_image,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region,
    size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
    size_t dst_slice_pitch, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, _cl_recorded_command **cmd,
    cl_command_buffer_khr command_buffer);

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
                              size_t size, cl_mem_object_type type, int *device_image_support, void *host_ptr,
                              cl_int *errcode_ret);

cl_int pocl_kernel_calc_wg_size (cl_command_queue command_queue,
                                 cl_kernel kernel, cl_uint work_dim,
                                 const size_t *global_work_offset,
                                 const size_t *global_work_size,
                                 const size_t *local_work_size,
                                 size_t *global_offset, size_t *local_size,
                                 size_t *num_groups);

cl_int pocl_kernel_collect_mem_objs (cl_command_queue command_queue,
                                     cl_kernel kernel, cl_uint *memobj_count,
                                     cl_mem *memobj_list,
                                     char *readonly_flag_list);

cl_int pocl_kernel_copy_args (cl_kernel kernel, _cl_command_run *command);

cl_int pocl_validate_copy_buffer (cl_command_queue command_queue,
                                  cl_mem src_buf, cl_mem dst_buf,
                                  size_t src_off, size_t dst_off, size_t size);

cl_int pocl_validate_copy_image (cl_mem src, cl_mem dst);

cl_int pocl_validate_fill_buffer (cl_command_queue command_queue,
                                  cl_mem buffer, const void *pattern,
                                  size_t pattern_size, size_t offset,
                                  size_t size);

cl_int pocl_validate_fill_image (cl_command_queue command_queue, cl_mem image,
                                 const void *fill_color, const size_t *origin,
                                 const size_t *region);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#define POCL_FILL_COMMAND_COPY_BUFFER                                         \
  do                                                                          \
    {                                                                         \
      cl_device_id dev = command_queue->device;                               \
      cmd->command.copy.src_mem_id                                            \
          = &src_buffer->device_ptrs[dev->global_mem_id];                     \
      cmd->command.copy.src_offset = src_offset;                              \
      cmd->command.copy.src = src_buffer;                                     \
      cmd->command.copy.dst_mem_id                                            \
          = &dst_buffer->device_ptrs[dev->global_mem_id];                     \
      cmd->command.copy.dst_offset = dst_offset;                              \
      cmd->command.copy.dst = dst_buffer;                                     \
      cmd->command.copy.size = size;                                          \
      if (src_buffer->size_buffer != NULL)                                    \
        {                                                                     \
          cmd->command.copy.src_content_size = src_buffer->size_buffer;       \
          cmd->command.copy.src_content_size_mem_id                           \
              = &src_buffer->size_buffer->device_ptrs[dev->dev_id];           \
        }                                                                     \
    }                                                                         \
  while (0)

#define POCL_FILL_COMMAND_COPY_BUFFER_RECT                                    \
  do                                                                          \
    {                                                                         \
      cl_device_id dev = command_queue->device;                               \
      cmd->command.copy_rect.src_mem_id                                       \
          = &src_buffer->device_ptrs[dev->global_mem_id];                     \
      cmd->command.copy_rect.src = src_buffer;                                \
      cmd->command.copy_rect.dst_mem_id                                       \
          = &dst_buffer->device_ptrs[dev->global_mem_id];                     \
      cmd->command.copy_rect.dst = dst_buffer;                                \
      cmd->command.copy_rect.src_origin[0] = src_offset + src_origin[0];      \
      cmd->command.copy_rect.src_origin[1] = src_origin[1];                   \
      cmd->command.copy_rect.src_origin[2] = src_origin[2];                   \
      cmd->command.copy_rect.dst_origin[0] = dst_offset + dst_origin[0];      \
      cmd->command.copy_rect.dst_origin[1] = dst_origin[1];                   \
      cmd->command.copy_rect.dst_origin[2] = dst_origin[2];                   \
      cmd->command.copy_rect.region[0] = region[0];                           \
      cmd->command.copy_rect.region[1] = region[1];                           \
      cmd->command.copy_rect.region[2] = region[2];                           \
    }                                                                         \
  while (0)

#define POCL_FILL_COMMAND_COPY_BUFFER_TO_IMAGE                                \
  do                                                                          \
    {                                                                         \
      cl_device_id dev = command_queue->device;                               \
      cmd->command.write_image.dst_mem_id                                     \
          = &dst_image->device_ptrs[dev->global_mem_id];                      \
      cmd->command.write_image.dst = dst_image;                               \
      cmd->command.write_image.src_host_ptr = NULL;                           \
      cmd->command.write_image.src_mem_id                                     \
          = &src_buffer->device_ptrs[dev->global_mem_id];                     \
      cmd->command.write_image.src = src_buffer;                              \
      cmd->command.write_image.src_row_pitch                                  \
          = 0; /* dst_image->image_row_pitch; */                              \
      cmd->command.write_image.src_slice_pitch                                \
          = 0; /* dst_image->image_slice_pitch; */                            \
      cmd->command.write_image.src_offset = src_offset;                       \
      cmd->command.write_image.origin[0] = dst_origin[0];                     \
      cmd->command.write_image.origin[1] = dst_origin[1];                     \
      cmd->command.write_image.origin[2] = dst_origin[2];                     \
      cmd->command.write_image.region[0] = region[0];                         \
      cmd->command.write_image.region[1] = region[1];                         \
      cmd->command.write_image.region[2] = region[2];                         \
    }                                                                         \
  while (0)

#define POCL_FILL_COMMAND_COPY_IMAGE                                          \
  do                                                                          \
    {                                                                         \
      cl_device_id dev = command_queue->device;                               \
      cmd->command.copy_image.src_mem_id                                      \
          = &src_image->device_ptrs[dev->global_mem_id];                      \
      cmd->command.copy_image.src = src_image;                                \
      cmd->command.copy_image.dst_mem_id                                      \
          = &dst_image->device_ptrs[dev->global_mem_id];                      \
      cmd->command.copy_image.dst = dst_image;                                \
      cmd->command.copy_image.src_origin[0] = src_origin[0];                  \
      cmd->command.copy_image.src_origin[1] = src_origin[1];                  \
      cmd->command.copy_image.src_origin[2] = src_origin[2];                  \
      cmd->command.copy_image.dst_origin[0] = dst_origin[0];                  \
      cmd->command.copy_image.dst_origin[1] = dst_origin[1];                  \
      cmd->command.copy_image.dst_origin[2] = dst_origin[2];                  \
      cmd->command.copy_image.region[0] = region[0];                          \
      cmd->command.copy_image.region[1] = region[1];                          \
      cmd->command.copy_image.region[2] = region[2];                          \
    }                                                                         \
  while (0)

#define POCL_FILL_COMMAND_COPY_IMAGE_TO_BUFFER                                \
  do                                                                          \
    {                                                                         \
      cl_device_id dev = command_queue->device;                               \
      cmd->command.read_image.src_mem_id                                      \
          = &src_image->device_ptrs[dev->global_mem_id];                      \
      cmd->command.read_image.src = src_image;                                \
      cmd->command.read_image.dst_host_ptr = NULL;                            \
      cmd->command.read_image.dst = dst_buffer;                               \
      cmd->command.read_image.dst_mem_id                                      \
          = &dst_buffer->device_ptrs[dev->global_mem_id];                     \
      cmd->command.read_image.origin[0] = src_origin[0];                      \
      cmd->command.read_image.origin[1] = src_origin[1];                      \
      cmd->command.read_image.origin[2] = src_origin[2];                      \
      cmd->command.read_image.region[0] = region[0];                          \
      cmd->command.read_image.region[1] = region[1];                          \
      cmd->command.read_image.region[2] = region[2];                          \
      cmd->command.read_image.dst_row_pitch                                   \
          = 0; /* src_image->image_row_pitch; */                              \
      cmd->command.read_image.dst_slice_pitch                                 \
          = 0; /* src_image->image_slice_pitch; */                            \
      cmd->command.read_image.dst_offset = dst_offset;                        \
    }                                                                         \
  while (0)

#define POCL_FILL_COMMAND_FILL_BUFFER                                         \
  do                                                                          \
    {                                                                         \
      cmd->command.memfill.dst_mem_id                                         \
          = &buffer->device_ptrs[command_queue->device->global_mem_id];       \
      cmd->command.memfill.size = size;                                       \
      cmd->command.memfill.offset = offset;                                   \
      void *p = pocl_aligned_malloc (pattern_size, pattern_size);             \
      memcpy (p, pattern, pattern_size);                                      \
      cmd->command.memfill.pattern = p;                                       \
      cmd->command.memfill.pattern_size = pattern_size;                       \
    }                                                                         \
  while (0)

#define POCL_FILL_COMMAND_FILL_IMAGE                                          \
  do                                                                          \
    {                                                                         \
      memcpy (cmd->command.fill_image.fill_pixel, fill_pattern, 16);          \
      cmd->command.fill_image.orig_pixel = fill_color_vec;                    \
      cmd->command.fill_image.pixel_size = px;                                \
      cmd->command.fill_image.mem_id                                          \
          = &image->device_ptrs[command_queue->device->global_mem_id];        \
      cmd->command.fill_image.origin[0] = origin[0];                          \
      cmd->command.fill_image.origin[1] = origin[1];                          \
      cmd->command.fill_image.origin[2] = origin[2];                          \
      cmd->command.fill_image.region[0] = region[0];                          \
      cmd->command.fill_image.region[1] = region[1];                          \
      cmd->command.fill_image.region[2] = region[2];                          \
    }                                                                         \
  while (0)

#define POCL_FILL_COMMAND_NDRANGEKERNEL                                       \
  do                                                                          \
    {                                                                         \
      cl_device_id realdev = pocl_real_dev (command_queue->device);           \
      for (unsigned i = 0; i < kernel->program->num_devices; ++i)             \
        {                                                                     \
          if (kernel->program->devices[i] == realdev)                         \
            program_dev_i = i;                                                \
        }                                                                     \
      assert (program_dev_i < CL_UINT_MAX);                                   \
      cmd->command.run.kernel = kernel;                                       \
      cmd->command.run.hash = kernel->meta->build_hash[program_dev_i];        \
      cmd->command.run.pc.local_size[0] = local[0];                           \
      cmd->command.run.pc.local_size[1] = local[1];                           \
      cmd->command.run.pc.local_size[2] = local[2];                           \
      cmd->command.run.pc.work_dim = work_dim;                                \
      cmd->command.run.pc.num_groups[0] = num_groups[0];                      \
      cmd->command.run.pc.num_groups[1] = num_groups[1];                      \
      cmd->command.run.pc.num_groups[2] = num_groups[2];                      \
      cmd->command.run.pc.global_offset[0] = offset[0];                       \
      cmd->command.run.pc.global_offset[1] = offset[1];                       \
      cmd->command.run.pc.global_offset[2] = offset[2];                       \
    }                                                                         \
  while (0)

#endif // POCL_SHARED_H
