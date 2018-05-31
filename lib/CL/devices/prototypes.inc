/* prototypes.inc - macros for generating the driver layer function prototypes

   Copyright (c) 2012-2016 Pekka Jääskeläinen / Tampere University of Technology
   
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

#ifdef _MSC_VER
#  define PUSH_VISIBILITY_HIDDEN
#  define POP_VISIBILITY_HIDDEN
#else
#  define PUSH_VISIBILITY_HIDDEN _Pragma ("GCC visibility push(hidden)")
#  define POP_VISIBILITY_HIDDEN _Pragma ("GCC visibility pop")
#endif

#define GEN_PROTOTYPES(__DRV__)                                         \
  PUSH_VISIBILITY_HIDDEN                               \
  void pocl_##__DRV__##_submit (_cl_command_node *node, cl_command_queue cq); \
  void pocl_##__DRV__##_join (cl_device_id device, cl_command_queue cq); \
  void pocl_##__DRV__##_flush (cl_device_id device, cl_command_queue cq); \
  void pocl_##__DRV__##_notify (cl_device_id device, cl_event event, cl_event finished);  \
  void pocl_##__DRV__##_broadcast (cl_event event);                    \
  void pocl_##__DRV__##_wait_event (cl_device_id device, cl_event event); \
  void pocl_##__DRV__##_update_event (cl_device_id device, cl_event event, cl_int status); \
  void pocl_##__DRV__##_free_event_data (cl_event event); \
  void pocl_##__DRV__##_init_device_ops(struct pocl_device_ops* ops);  \
  cl_int pocl_##__DRV__##_uninit (cl_device_id device);                   \
  cl_int pocl_##__DRV__##_reinit (cl_device_id device);                   \
  cl_int pocl_##__DRV__##_init (unsigned j, cl_device_id device, const char* parameters); \
  unsigned int pocl_##__DRV__##_probe (struct pocl_device_ops *ops); \
  cl_int pocl_##__DRV__##_init_queue (cl_command_queue queue); \
  void pocl_##__DRV__##_free_queue (cl_command_queue queue); \
  cl_int pocl_##__DRV__##_alloc_mem_obj (cl_device_id device, cl_mem mem_obj, \
                                         void* host_ptr); \
  void *pocl_##__DRV__##_create_sub_buffer (void *device_data, void* buffer, \
                                            size_t origin, size_t size); \
  void pocl_##__DRV__##_free (cl_device_id device, cl_mem mem_obj);   \
  void pocl_##__DRV__##_free_ptr (cl_device_id device, void* mem_ptr);   \
  void pocl_##__DRV__##_read (void *data, void *host_ptr,                   \
                          const void *device_ptr, size_t offset, size_t cb); \
  void pocl_##__DRV__##_read_rect (void *data, void *host_ptr,          \
                               void *device_ptr,                        \
                               const size_t *buffer_origin,             \
                               const size_t *host_origin,               \
                               const size_t *region,                    \
                               size_t buffer_row_pitch,                 \
                               size_t buffer_slice_pitch,               \
                               size_t host_row_pitch,                   \
                               size_t host_slice_pitch);                \
  void pocl_##__DRV__##_write (void *data, const void *host_ptr,        \
                               void *device_ptr, size_t offset, size_t cb); \
  void pocl_##__DRV__##_write_rect (void *data, const void *host_ptr,  \
                                void *device_ptr,                       \
                                const size_t *buffer_origin,            \
                                const size_t *host_origin,              \
                                const size_t *region,                   \
                                size_t buffer_row_pitch,                \
                                size_t buffer_slice_pitch,              \
                                size_t host_row_pitch,                  \
                                size_t host_slice_pitch);               \
  void pocl_##__DRV__##_copy (void *data, const void *src_ptr, \
                             size_t src_offset, void *__restrict__ dst_ptr, \
                             size_t dst_offset, size_t cb);       \
  void pocl_##__DRV__##_copy_rect (void *data, const void *src_ptr,         \
                               void *dst_ptr,                           \
                               const size_t *src_origin,                \
                               const size_t *dst_origin,                \
                               const size_t *region,                    \
                               size_t src_row_pitch,                    \
                               size_t src_slice_pitch,                  \
                               size_t dst_row_pitch,                    \
                               size_t dst_slice_pitch);                 \
  void pocl_##__DRV__##_fill_rect (void *data,           \
                           void *__restrict__ const device_ptr, \
                           const size_t *__restrict__ const buffer_origin,  \
                           const size_t *__restrict__ const region,         \
                           size_t const buffer_row_pitch,      \
                           size_t const buffer_slice_pitch,    \
                           void *fill_pixel,    \
                           size_t pixel_size);  \
  void pocl_##__DRV__##_compile_kernel (_cl_command_node *node, \
                                        cl_kernel kernel, \
                                        cl_device_id device);  \
  void pocl_##__DRV__##_memfill (void *ptr,           \
                            size_t size,              \
                            size_t offset,            \
                            const void* pattern,      \
                            size_t pattern_size);     \
  void pocl_##__DRV__##_run (void *data, _cl_command_node* cmd);        \
  void pocl_##__DRV__##_run_native (void *data, _cl_command_node* cmd); \
  void* pocl_##__DRV__##_map_mem (void *data, void *buf_ptr,         \
                        size_t offset, size_t size, void *host_ptr); \
  void* pocl_##__DRV__##_unmap_mem (void *data, void *host_ptr, \
                                    void *device_start_ptr, size_t offset, \
                                    size_t size); \
  cl_ulong pocl_##__DRV__##_get_timer_value(void *data); \
  char* pocl_##__DRV__##_init_build (void *data); \
  char* pocl_##__DRV__##_build_hash (cl_device_id device); \
  void pocl_##__DRV__##_init_target_machine (void *data, void* target_machine);\
  cl_int pocl_##__DRV__##_get_supported_image_formats (cl_mem_flags flags,\
                                         const cl_image_format **image_formats,\
                                         cl_uint *num_image_formats);\

POP_VISIBILITY_HIDDEN
