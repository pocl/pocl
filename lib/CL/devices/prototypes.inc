/* prototypes.inc - macros for generating the driver layer function prototypes

   Copyright (c) 2012-2019 Pekka Jääskeläinen
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#define GEN_PROTOTYPES(__DRV__)                                               \
  void pocl_##__DRV__##_submit (_cl_command_node *node, cl_command_queue cq); \
  void pocl_##__DRV__##_join (cl_device_id device, cl_command_queue cq);      \
  void pocl_##__DRV__##_flush (cl_device_id device, cl_command_queue cq);     \
  void pocl_##__DRV__##_notify (cl_device_id device, cl_event event,          \
                                cl_event finished);                           \
  void pocl_##__DRV__##_broadcast (cl_event event);                           \
  void pocl_##__DRV__##_wait_event (cl_device_id device, cl_event event);     \
  void pocl_##__DRV__##_update_event (cl_device_id device, cl_event event);   \
  void pocl_##__DRV__##_free_event_data (cl_event event);                     \
  POCL_EXPORT                                                                 \
  void pocl_##__DRV__##_init_device_ops (struct pocl_device_ops *ops);        \
  cl_int pocl_##__DRV__##_uninit (unsigned j, cl_device_id device);           \
  cl_int pocl_##__DRV__##_reinit (unsigned j, cl_device_id device,            \
                                  const char *parameters);                    \
  cl_int pocl_##__DRV__##_init (unsigned j, cl_device_id device,              \
                                const char *parameters);                      \
  unsigned int pocl_##__DRV__##_probe (struct pocl_device_ops *ops);          \
  cl_int pocl_##__DRV__##_alloc_mem_obj (cl_device_id device, cl_mem mem_obj, \
                                         void *host_ptr);                     \
  cl_int pocl_##__DRV__##_alloc_subbuffer (cl_device_id device, cl_mem sub_buf); \
  void pocl_##__DRV__##_free (cl_device_id device, cl_mem mem_obj);           \
  int pocl_##__DRV__##_can_migrate_d2d (cl_device_id dest,                    \
                                        cl_device_id source);                 \
  int pocl_##__DRV__##_migrate_d2d (                                          \
      cl_device_id src_dev, cl_device_id dst_dev, cl_mem mem,                 \
      pocl_mem_identifier *src_mem_id, pocl_mem_identifier *dst_mem_id);      \
  void pocl_##__DRV__##_read (void *data, void *__restrict__ dst_host_ptr,    \
                              pocl_mem_identifier *src_mem_id,                \
                              cl_mem src_buf, size_t offset, size_t size);    \
  void pocl_##__DRV__##_read_rect (                                           \
      void *data, void *__restrict__ dst_host_ptr,                            \
      pocl_mem_identifier *src_mem_id, cl_mem src_buf,                        \
      const size_t *buffer_origin, const size_t *host_origin,                 \
      const size_t *region, size_t buffer_row_pitch,                          \
      size_t buffer_slice_pitch, size_t host_row_pitch,                       \
      size_t host_slice_pitch);                                               \
  void pocl_##__DRV__##_write (void *data,                                    \
                               const void *__restrict__ src_host_ptr,         \
                               pocl_mem_identifier *dst_mem_id,               \
                               cl_mem dst_buf, size_t offset, size_t size);   \
  void pocl_##__DRV__##_write_rect (                                          \
      void *data, const void *__restrict__ src_host_ptr,                      \
      pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,                        \
      const size_t *buffer_origin, const size_t *host_origin,                 \
      const size_t *region, size_t buffer_row_pitch,                          \
      size_t buffer_slice_pitch, size_t host_row_pitch,                       \
      size_t host_slice_pitch);                                               \
  void pocl_##__DRV__##_copy (                                                \
      void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,            \
      pocl_mem_identifier *src_mem_id, cl_mem src_buf,                        \
      size_t dst_offset, size_t src_offset, size_t size);                     \
  void pocl_##__DRV__##_copy_with_size (                                      \
      void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,            \
      pocl_mem_identifier *src_mem_id, cl_mem src_buf,                        \
      pocl_mem_identifier *content_size_buf_mem_id, cl_mem content_size_buf,  \
      size_t dst_offset, size_t src_offset, size_t size);                     \
  void pocl_##__DRV__##_copy_rect (                                           \
      void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,            \
      pocl_mem_identifier *src_mem_id, cl_mem src_buf,                        \
      const size_t *dst_origin, const size_t *src_origin,                     \
      const size_t *region, size_t dst_row_pitch, size_t dst_slice_pitch,     \
      size_t src_row_pitch, size_t src_slice_pitch);                          \
  void pocl_##__DRV__##_compile_kernel (_cl_command_node *node,               \
                                        cl_kernel kernel,                     \
                                        cl_device_id device, int specialize); \
  int pocl_##__DRV__##_init_queue (cl_device_id device,                       \
                                   cl_command_queue queue);                   \
  int pocl_##__DRV__##_free_queue (cl_device_id device,                       \
                                   cl_command_queue queue);                   \
  int pocl_##__DRV__##_init_context (cl_device_id device,                     \
                                     cl_context context);                     \
  int pocl_##__DRV__##_free_context (cl_device_id device,                     \
                                     cl_context context);                     \
  int pocl_##__DRV__##_create_kernel (cl_device_id device, cl_program p,      \
                                      cl_kernel k, unsigned device_i);        \
  int pocl_##__DRV__##_free_kernel (cl_device_id device, cl_program p,        \
                                    cl_kernel k, unsigned device_i);          \
  int pocl_##__DRV__##_build_source (                                         \
      cl_program program, cl_uint device_i, cl_uint num_input_headers,        \
      const cl_program *input_headers, const char **header_include_names,     \
      int link_program);                                                      \
  int pocl_##__DRV__##_build_binary ( cl_program program,                     \
      cl_uint device_i, int link_program, int spir_build);                    \
  int pocl_##__DRV__##_build_builtin (cl_program program, cl_uint device_i);  \
  int pocl_##__DRV__##_build_defined_builtin (cl_program program, cl_uint device_i);  \
  int pocl_##__DRV__##_link_program (                                         \
      cl_program program, cl_uint device_i, cl_uint num_input_programs,       \
      const cl_program *input_programs, int create_library);                  \
  int pocl_##__DRV__##_post_build_program (cl_program program,                \
                                           cl_uint device_i);                 \
  int pocl_##__DRV__##_build_poclbinary (cl_program program,                  \
                                         cl_uint device_i);                   \
  int pocl_##__DRV__##_free_program (cl_device_id device, cl_program program, \
                                     unsigned program_device_i);              \
  int pocl_##__DRV__##_setup_metadata (                                       \
      cl_device_id device, cl_program program, unsigned program_device_i);    \
  int pocl_##__DRV__##_supports_binary (                                      \
      cl_device_id device, size_t length, const char *binary);                \
  int pocl_##__DRV__##_supports_dbk (cl_device_id device,                     \
                                     BuiltinKernelId kernel_id,               \
                                     const void *kernel_attributes);          \
  void pocl_##__DRV__##_memfill (void *data, pocl_mem_identifier *dst_mem_id, \
                                 cl_mem dst_buf, size_t size, size_t offset,  \
                                 const void *__restrict__ pattern,            \
                                 size_t pattern_size);                        \
  void pocl_##__DRV__##_run (void *data, _cl_command_node *cmd);              \
  void pocl_##__DRV__##_run_native (void *data, _cl_command_node *cmd);       \
  cl_int pocl_##__DRV__##_map_mem (void *data,                                \
                                   pocl_mem_identifier *src_mem_id,           \
                                   cl_mem src_buf, mem_mapping_t *map);       \
  cl_int pocl_##__DRV__##_unmap_mem (void *data,                              \
                                     pocl_mem_identifier *dst_mem_id,         \
                                     cl_mem dst_buf, mem_mapping_t *map);     \
  cl_int pocl_##__DRV__##_get_mapping_ptr (void *data,                        \
                            pocl_mem_identifier *mem_id,                      \
                            cl_mem mem, mem_mapping_t *map);                  \
  cl_int pocl_##__DRV__##_free_mapping_ptr (void *data,                       \
                            pocl_mem_identifier *mem_id,                      \
                            cl_mem mem, mem_mapping_t *map);                  \
  char *pocl_##__DRV__##_init_build (void *data);                             \
  char *pocl_##__DRV__##_build_hash (cl_device_id device);                    \
  void pocl_##__DRV__##_init_target_machine (void *data,                      \
                                             void *target_machine);           \
  int pocl_##__DRV__##_create_sampler (cl_device_id device, cl_sampler samp,  \
                                       unsigned device_i);                    \
  int pocl_##__DRV__##_free_sampler (cl_device_id device, cl_sampler samp,    \
                                     unsigned device_i);                      \
  cl_int pocl_##__DRV__##_copy_image_rect (                                   \
      void *data, cl_mem src_image, cl_mem dst_image,                         \
      pocl_mem_identifier *src_mem_id, pocl_mem_identifier *dst_mem_id,       \
      const size_t *src_origin, const size_t *dst_origin,                     \
      const size_t *region);                                                  \
  cl_int pocl_##__DRV__##_write_image_rect (                                  \
      void *data, cl_mem dst_image, pocl_mem_identifier *dst_mem_id,          \
      const void *__restrict__ src_host_ptr, pocl_mem_identifier *src_mem_id, \
      const size_t *origin, const size_t *region, size_t src_row_pitch,       \
      size_t src_slice_pitch, size_t src_offset);                             \
  cl_int pocl_##__DRV__##_read_image_rect (                                   \
      void *data, cl_mem src_image, pocl_mem_identifier *src_mem_id,          \
      void *__restrict__ dst_host_ptr, pocl_mem_identifier *dst_mem_id,       \
      const size_t *origin, const size_t *region, size_t dst_row_pitch,       \
      size_t dst_slice_pitch, size_t dst_offset);                             \
  cl_int pocl_##__DRV__##_map_image (void *data, pocl_mem_identifier *mem_id, \
                                     cl_mem src_image, mem_mapping_t *map);   \
  cl_int pocl_##__DRV__##_unmap_image (void *data,                            \
                                       pocl_mem_identifier *mem_id,           \
                                       cl_mem dst_image, mem_mapping_t *map); \
  cl_int pocl_##__DRV__##_fill_image (                                        \
      void *data, cl_mem image, pocl_mem_identifier *mem_id,                  \
      const size_t *origin, const size_t *region,                             \
      cl_uint4 orig_pixel, pixel_t fill_pixel,  size_t pixel_size);           \
  cl_int pocl_##__DRV__##_get_gl_context_assoc                                \
         (cl_device_id device,              \
          cl_gl_context_info type,                  \
          const cl_context_properties *properties);      \
  void pocl_##__DRV__##_svm_free (cl_device_id dev, void *svm_ptr);           \
  void *pocl_##__DRV__##_svm_alloc (cl_device_id dev, cl_svm_mem_flags flags, \
                                    size_t size);                             \
  void pocl_##__DRV__##_svm_map (cl_device_id dev, void *svm_ptr);            \
  void pocl_##__DRV__##_svm_unmap (cl_device_id dev, void *svm_ptr);          \
  void pocl_##__DRV__##_svm_copy (cl_device_id dev, void *__restrict__ dst,   \
                                  const void *__restrict__ src, size_t size); \
  void pocl_##__DRV__##_svm_fill (                                            \
      cl_device_id dev, void *__restrict__ svm_ptr, size_t size,              \
      void *__restrict__ pattern, size_t pattern_size);                       \
  void pocl_##__DRV__##_svm_migrate (cl_device_id dev,                        \
        size_t num_svm_pointers,                                              \
                       void *__restrict__ svm_pointers,    \
                       size_t *__restrict__ sizes); \
  void pocl_##__DRV__##_svm_register (cl_device_id dev, void *host_ptr, size_t size);      \
  void pocl_##__DRV__##_svm_unregister (cl_device_id dev, void *host_ptr, size_t size);    \
  void pocl_##__DRV__##_svm_copy_rect (cl_device_id dev, \
                                       void *__restrict__ dst,  \
                                       const void *__restrict__ src,  \
                                       const size_t *dst_origin, \
                                       const size_t *src_origin, \
                                       const size_t *region, \
                                       size_t dst_row_pitch, \
                                       size_t dst_slice_pitch,  \
                                       size_t src_row_pitch,\
                                       size_t src_slice_pitch); \
  void pocl_##__DRV__##_svm_fill_rect (cl_device_id dev, \
                                       void *__restrict__ svm_ptr, \
                                       const size_t *origin, \
                                       const size_t *region, \
                                       size_t row_pitch, \
                                       size_t slice_pitch, \
                                       void *__restrict__ pattern, \
                                       size_t pattern_size); \
  void pocl_##__DRV__##_notify_cmdq_finished (cl_command_queue cq);           \
  void pocl_##__DRV__##_notify_event_finished (cl_event event);               \
  cl_int pocl_##__DRV__##_get_device_info_ext(cl_device_id dev,               \
                                         cl_device_info param_name,           \
                                         size_t param_value_size,             \
                                         void * param_value,                  \
                                         size_t * param_value_size_ret);      \
  cl_int pocl_##__DRV__##_get_subgroup_info_ext (cl_device_id dev, cl_kernel kernel, \
                                         unsigned program_device_i, \
                                         cl_kernel_sub_group_info param_name, \
                                         size_t input_value_size, const void *input_value, \
                                         size_t param_value_size, void *param_value, \
                                         size_t *param_value_size_ret); \
cl_int pocl_##__DRV__##_set_kernel_exec_info_ext (cl_device_id dev,          \
                                      unsigned program_device_i, \
                                      cl_kernel kernel,   \
                                      cl_uint param_name,  \
                                      size_t param_value_size,  \
                                      const void * param_value);  \
  cl_int pocl_##__DRV__##_get_synchronized_timestamps (cl_device_id dev,     \
                                         cl_ulong *dev_timestamp,            \
                                         cl_ulong *host_timestamp);          \
  void pocl_##__DRV__##_usm_free (cl_device_id dev, \
                                  void *svm_ptr); \
  void pocl_##__DRV__##_usm_free_blocking (cl_device_id dev, \
                                           void *svm_ptr); \
  void * pocl_##__DRV__##_usm_alloc (cl_device_id dev, unsigned alloc_type, \
                                     cl_mem_alloc_flags_intel flags, size_t size, \
                                     cl_int *errcode);

