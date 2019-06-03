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

cl_program create_program_skeleton (cl_context context, cl_uint num_devices,
                                    const cl_device_id *device_list,
                                    const size_t *lengths,
                                    const unsigned char **binaries,
                                    cl_int *binary_status, cl_int *errcode_ret,
                                    int allow_empty_binaries);

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

int context_set_properties(cl_context                    context,
                           const cl_context_properties * properties,
                           cl_int *                      errcode);

cl_mem pocl_create_memobject (cl_context context, cl_mem_flags flags,
                              size_t size, cl_mem_object_type type, int *device_image_support, void *host_ptr,
                              cl_int *errcode_ret);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif // POCL_SHARED_H
