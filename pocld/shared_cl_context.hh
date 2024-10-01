/* shared_cl_context.hh - pocld class that wraps an OCL context and resources

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2023 Jan Solanti / Tampere University
   Copyright (c) 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <CL/opencl.hpp>

#include "common.hh"

#ifndef POCL_REMOTE_SHARED_CL_HH
#define POCL_REMOTE_SHARED_CL_HH

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class SharedContextBase {

public:
  virtual ~SharedContextBase() {}

  virtual cl::Context getHandle() const = 0;

  virtual void queuedPush(Request *req) = 0;

  virtual void notifyEvent(uint64_t id, cl_int status) = 0;

  virtual bool isCommandReceived(uint64_t id) = 0;

  virtual size_t numDevices() const = 0;

  virtual int writeKernelMeta(uint32_t program_id, char *buffer,
                              size_t *written) = 0;

  virtual EventPair getEventPairForId(uint64_t event_id) = 0;

  virtual int waitAndDeleteEvent(uint64_t event_id) = 0;

  virtual std::vector<cl::Event> remapWaitlist(size_t num_events, uint64_t *ids,
                                               uint64_t dep) = 0;

#ifdef ENABLE_RDMA
  virtual bool clientUsesRdma() = 0;

  virtual char *getRdmaShadowPtr(uint32_t id) = 0;
#endif

  /************************************************************************/

  virtual int createBuffer(BufferId_t BufferID, size_t Size, uint64_t Flags,
                           void *HostPtr, BufferId_t ParentID, size_t Origin,
                           void **DeviceAddr) = 0;

  virtual int freeBuffer(BufferId_t BufferID, bool is_svm) = 0;

  virtual int buildOrLinkProgram(
      uint32_t program_id, std::vector<uint32_t> &DeviceList, char *source,
      size_t source_size, bool is_binary, bool is_builtin, bool is_spirv,
      const char *options,
      std::unordered_map<uint64_t, std::vector<unsigned char>> &input_binaries,
      std::unordered_map<uint64_t, std::vector<unsigned char>> &output_binaries,
      std::unordered_map<uint64_t, std::string> &build_logs,
      size_t &num_kernels, uint64_t SVMRegionOffset, bool CompileOnly,
      bool LinkOnly) = 0;

  virtual int freeProgram(uint32_t program_id) = 0;

  virtual int createKernel(uint32_t kernel_id, uint32_t program_id,
                           const char *name) = 0;

  virtual int freeKernel(uint32_t kernel_id) = 0;

  virtual int createQueue(uint32_t queue_id, uint32_t dev_id) = 0;

  virtual int freeQueue(uint32_t queue_id) = 0;

  virtual int getDeviceInfo(uint32_t device_id, DeviceInfo_t &i,
                            std::vector<std::string> &strings,
                            cl_device_info SpecificInfo) = 0;

  virtual int createSampler(uint32_t sampler_id, uint32_t normalized,
                            uint32_t address, uint32_t filter) = 0;

  virtual int freeSampler(uint32_t sampler_id) = 0;

  virtual int createImage(uint32_t image_id, uint32_t flags,
                          // format
                          uint32_t channel_order, uint32_t channel_data_type,
                          // desc
                          uint32_t type, uint32_t width, uint32_t height,
                          uint32_t depth, uint32_t array_size,
                          uint32_t row_pitch, uint32_t slice_pitch) = 0;

  virtual int freeImage(uint32_t image_id) = 0;

  /************************************************************************/

  virtual int migrateMemObject(uint64_t ev_id, uint32_t cq_id,
                               uint32_t mem_obj_id, unsigned is_image,
                               EventTiming_t &evt, uint32_t waitlist_size,
                               uint64_t *waitlist) = 0;

  /**********************************************************************/
  /**********************************************************************/
  /**********************************************************************/

  virtual int readBuffer(uint64_t ev_id, uint32_t cq_id, uint64_t buffer_id,
                         int is_svm, uint32_t size_id, size_t size,
                         size_t offset, void *host_ptr, uint64_t *content_size,
                         EventTiming_t &evt, uint32_t waitlist_size,
                         uint64_t *waitlist) = 0;

  virtual int writeBuffer(uint64_t ev_id, uint32_t cq_id, uint64_t buffer_id,
                          int is_svm, size_t size, size_t offset,
                          void *host_ptr, EventTiming_t &evt,
                          uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int copyBuffer(uint64_t ev_id, uint32_t cq_id, uint32_t src_buffer_id,
                         uint32_t dst_buffer_id,
                         uint32_t content_size_buffer_id, size_t size,
                         size_t src_offset, size_t dst_offset,
                         EventTiming_t &evt, uint32_t waitlist_size,
                         uint64_t *waitlist) = 0;

  virtual int readBufferRect(uint64_t ev_id, uint32_t cq_id, uint32_t buffer_id,
                             sizet_vec3 &buffer_origin, sizet_vec3 &region,
                             size_t buffer_row_pitch, size_t buffer_slice_pitch,
                             void *host_ptr, size_t host_bytes,
                             EventTiming_t &evt, uint32_t waitlist_size,
                             uint64_t *waitlist) = 0;

  virtual int writeBufferRect(uint64_t ev_id, uint32_t cq_id,
                              uint32_t buffer_id, sizet_vec3 &buffer_origin,
                              sizet_vec3 &region, size_t buffer_row_pitch,
                              size_t buffer_slice_pitch, void *host_ptr,
                              size_t host_bytes, EventTiming_t &evt,
                              uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int copyBufferRect(uint64_t ev_id, uint32_t cq_id,
                             uint32_t dst_buffer_id, uint32_t src_buffer_id,
                             sizet_vec3 &dst_origin, sizet_vec3 &src_origin,
                             sizet_vec3 &region, size_t dst_row_pitch,
                             size_t dst_slice_pitch, size_t src_row_pitch,
                             size_t src_slice_pitch, EventTiming_t &evt,
                             uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int fillBuffer(uint64_t ev_id, uint32_t cq_id, uint32_t buffer_id,
                         size_t offset, size_t size, void *pattern,
                         size_t pattern_size, EventTiming_t &evt,
                         uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int runKernel(uint64_t ev_id, uint32_t cq_id, uint32_t device_id,
                        uint16_t has_new_args, size_t arg_count, uint64_t *args,
                        unsigned char *is_svm_ptr, size_t pod_size,
                        char *pod_buf, EventTiming_t &evt, uint32_t kernel_id,
                        uint32_t waitlist_size, uint64_t *waitlist,
                        unsigned dim, const sizet_vec3 &offset,
                        const sizet_vec3 &global,
                        const sizet_vec3 *local = nullptr) = 0;

  /**********************************************************************/
  /**********************************************************************/
  /**********************************************************************/

  virtual int fillImage(uint64_t ev_id, uint32_t cq_id, uint32_t image_id,
                        sizet_vec3 &origin, sizet_vec3 &region,
                        void *fill_color, EventTiming_t &evt,
                        uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int copyImage2Buffer(uint64_t ev_id, uint32_t cq_id,
                               uint32_t image_id, uint32_t dst_buf_id,
                               sizet_vec3 &origin, sizet_vec3 &region,
                               size_t offset, EventTiming_t &evt,
                               uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int copyBuffer2Image(uint64_t ev_id, uint32_t cq_id,
                               uint32_t image_id, uint32_t src_buf_id,
                               sizet_vec3 &origin, sizet_vec3 &region,
                               size_t offset, EventTiming_t &evt,
                               uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int copyImage2Image(uint64_t ev_id, uint32_t cq_id,
                              uint32_t dst_image_id, uint32_t src_image_id,
                              sizet_vec3 &dst_origin, sizet_vec3 &src_origin,
                              sizet_vec3 &region, EventTiming_t &evt,
                              uint32_t waitlist_size, uint64_t *waitlist) = 0;

  virtual int readImageRect(uint64_t ev_id, uint32_t cq_id, uint32_t image_id,
                            sizet_vec3 &origin, sizet_vec3 &region,
                            void *host_ptr, size_t host_bytes,
                            EventTiming_t &evt, uint32_t waitlist_size,
                            uint64_t *waitlist) = 0;

  virtual int writeImageRect(uint64_t ev_id, uint32_t cq_id, uint32_t image_id,
                             sizet_vec3 &origin, sizet_vec3 &region,
                             void *host_ptr, size_t host_bytes,
                             EventTiming_t &evt, uint32_t waitlist_size,
                             uint64_t *waitlist) = 0;
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
