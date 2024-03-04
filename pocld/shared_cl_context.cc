/* shared_cl_context.cc - pocld class that wraps an OCL context and resources

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2023 Jan Solanti / Tampere University
   Copyright (c) 2023 Pekka Jääskeläinen / Intel Finland Oy

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

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "bufalloc.h"
#include "cmd_queue.hh"
#include "common.hh"
#include "config.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"
#include "shared_cl_context.hh"
#include "spirv_parser.hh"
#include "virtual_cl_context.hh"

#define EVENT_TIMING_PRE                                                       \
  cl::Event event{};                                                           \
  int err = 0;

#define EVENT_TIMING_POST(msg)                                                 \
  {                                                                            \
    std::unique_lock<std::mutex> lock(EventmapMutex);                          \
    auto map_result = Eventmap.insert({ev_id, {event, cl::UserEvent()}});      \
    if (!map_result.second) {                                                  \
      assert(!map_result.first->second.native.get());                          \
      map_result.first->second.native = event;                                 \
    }                                                                          \
  }                                                                            \
  if (err == CL_SUCCESS)                                                       \
    POCL_MSG_PRINT_EVENTS(msg " event %" PRIu64 "\n", ev_id);                  \
  else {                                                                       \
    POCL_MSG_ERR("%s = %d, event %" PRIu64 "\n", msg, err, ev_id);             \
  }                                                                            \
  return err;

#define EVENT_TIMING(msg, code)                                                \
  EVENT_TIMING_PRE;                                                            \
  err = code;                                                                  \
  EVENT_TIMING_POST(msg)

/****************************************************************************************************************/
/****************************************************************************************************************/
/****************************************************************************************************************/

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

const static sizet_vec3 zero_origin = {0, 0, 0};

#define CHECK_IMAGE_SUPPORT()                                                  \
  if (!hasImageSupport) {                                                      \
    POCL_MSG_PRINT_GENERAL("Context has no image support, return\n");          \
    return CL_SUCCESS;                                                         \
  }

class SharedCLContext final : public SharedContextBase {
  cl::Context ContextWithAllDevices;
  cl::Context ContextWithSVMDevices;

  std::vector<cl::Device> CLDevices;
  std::vector<cl::Device> CLDevicesWithSVMSupport;

  std::unordered_map<uint32_t, clSamplerPtr> SamplerIDmap;
  std::unordered_map<uint32_t, clImagePtr> ImageIDmap;
  std::unordered_map<uint32_t, clProgramStructPtr> ProgramIDmap;
  std::unordered_map<uint32_t, clKernelStructPtr> KernelIDmap;
  std::unordered_map<uint32_t, clCommandQueuePtr> QueueIDMap;

  std::unordered_map<BufferId_t, clBufferPtr> BufferIDmap;
  std::unordered_map<BufferId_t, void *> SVMBackingStoreMap;
  // Index for finding the buffer ids of shadow subbuffers of SVM
  // chunks.
  std::unordered_map<void *, BufferId_t> SVMShadowBufferIDMap;

  // A mutex guarding the access to the above two buffer maps.
  // TODO: check that the more fine grained mutex suffices here.
  std::mutex BufferMapMutex;

  std::mutex EventmapMutex;
  std::unordered_map<uint64_t, EventPair> Eventmap;

  std::mutex MainMutex;
  // threads
  std::unordered_map<uint32_t, CommandQueueUPtr> QueueThreadMap;

  ReplyQueueThread *slow, *fast;

  VirtualContextBase *ParentCtx;
  unsigned plat_id;
  bool hasImageSupport;
  std::string name;

  struct SVMRegion {
    // The size and start of the memory region.
    size_t Size;
    void *StartAddress;

    // Bufalloc memory book keeping for allocations from the SVM pool.
    memory_region_t *Allocations;

    // The parent cl_mem wrapping the entire SVM region from which
    // subbuffers are internally allocated for client buffer requests.
    cl_mem ShadowBuffer;
  };

  // All SVM regions from which we can allocate memory. We have
  // multiple regions since clSVMAlloc() is limited by the max allocation
  // size which is typically much less than the total allocatable SVM (global
  // mem) on the device. If the size of the SVMRegions is non zero, all
  // allocations, including the cl_mem allocations will be allocated from a
  // contiguous region in one of the regions.
  std::vector<SVMRegion> SVMRegions;
  // The smallest starting address of an SVM region. This will be the "base" of
  // the large "virtual" SVM region.
  void *SVMRegionsStartAddress = nullptr;
  // The first (nonallocatable) address after the virtual SVM region formed by
  // multiple SVM allocations.
  void *SVMRegionsEndAddress = nullptr;

  // How much waste space is allowed between the regions. The waste space is
  // caused by clSVMAlloc() not returning the regions right after each other,
  // but potentially anywhere from VM, leaving unallocatable gaps between the
  // regions. Annoyingly this allocation scheme might lead to different sized
  // SVM regions for each run, depending how the device's driver allocates its
  // SVM to the server VM. 16 GB of waste should be reasonable for 64b host
  // system.
  const size_t SVMMaxAllowedWasteSpace = (size_t)16 * 1024 * 1024 * 1024;

  int setKernelArgs(cl::Kernel *k, clKernelStruct *kernel, size_t arg_count,
                    uint64_t *args, unsigned char *is_svm_ptr, size_t pod_size,
                    char *pod_buf);

public:
  SharedCLContext(cl::Platform *p, unsigned plat_id, VirtualContextBase *v,
                  ReplyQueueThread *s, ReplyQueueThread *f);

  virtual ~SharedCLContext();

  virtual cl::Context getHandle() const override {
    return ContextWithAllDevices;
  }

  virtual size_t numDevices() const override { return CLDevices.size(); }

  virtual void queuedPush(Request *req) override;

  virtual void notifyEvent(uint64_t id, cl_int status) override;

  virtual bool isCommandReceived(uint64_t id) override;

  virtual int writeKernelMeta(uint32_t program_id, char *buffer,
                              size_t *written) override;

  virtual EventPair getEventPairForId(uint64_t event_id) override;

  virtual int waitAndDeleteEvent(uint64_t event_id) override;

  virtual std::vector<cl::Event> remapWaitlist(size_t num_events, uint64_t *ids,
                                               uint64_t dep) override;

#ifdef ENABLE_RDMA
  virtual bool clientUsesRdma() override {
    return ParentCtx->clientUsesRdma();
  };

  virtual char *getRdmaShadowPtr(uint32_t id) override {
    return ParentCtx->getRdmaShadowPtr(id);
  };
#endif

  /************************************************************************/

  virtual int createBuffer(BufferId_t BufferID, size_t Size, uint64_t Flags,
                           void *HostPtr, void **DeviceAddr) override;

  virtual int freeBuffer(uint64_t buffer_id, bool is_svm) override;

  virtual int buildOrLinkProgram(
      uint32_t program_id, std::vector<uint32_t> &DeviceList, char *source,
      size_t source_size, bool is_binary, bool is_builtin, bool is_spirv,
      const char *options,
      std::unordered_map<uint64_t, std::vector<unsigned char>> &input_binaries,
      std::unordered_map<uint64_t, std::vector<unsigned char>> &output_binaries,
      std::unordered_map<uint64_t, std::string> &build_logs,
      size_t &num_kernels, uint64_t svm_region_offset, bool CompileOnly = false,
      bool LinkOnly = false) override;

  virtual int freeProgram(uint32_t program_id) override;

  virtual int createKernel(uint32_t kernel_id, uint32_t program_id,
                           const char *name) override;

  virtual int freeKernel(uint32_t kernel_id) override;

  virtual int createQueue(uint32_t queue_id, uint32_t dev_id) override;

  virtual int freeQueue(uint32_t queue_id) override;

  virtual int getDeviceInfo(uint32_t device_id, DeviceInfo_t &i,
                            std::vector<std::string>& strings) override;

  virtual int createSampler(uint32_t sampler_id, uint32_t normalized,
                            uint32_t address, uint32_t filter) override;

  virtual int freeSampler(uint32_t sampler_id) override;

  virtual int createImage(uint32_t image_id, uint32_t flags,
                          // format
                          uint32_t channel_order, uint32_t channel_data_type,
                          // desc
                          uint32_t type, uint32_t width, uint32_t height,
                          uint32_t depth, uint32_t array_size,
                          uint32_t row_pitch, uint32_t slice_pitch) override;

  virtual int freeImage(uint32_t image_id) override;

  /**************************************************************************/

  virtual int migrateMemObject(uint64_t ev_id, uint32_t cq_id,
                               uint32_t mem_obj_id, unsigned is_image,
                               EventTiming_t &evt, uint32_t waitlist_size,
                               uint64_t *waitlist) override;

  /**********************************************************************/
  /**********************************************************************/
  /**********************************************************************/

  virtual int readBuffer(uint64_t ev_id, uint32_t cq_id, uint64_t buffer_id,
                         int is_svm, uint32_t size_id, size_t size,
                         size_t offset, void *host_ptr, uint64_t *content_size,
                         EventTiming_t &evt, uint32_t waitlist_size,
                         uint64_t *waitlist) override;

  virtual int writeBuffer(uint64_t ev_id, uint32_t cq_id, uint64_t buffer_id,
                          int is_svm, size_t size, size_t offset,
                          void *host_ptr, EventTiming_t &evt,
                          uint32_t waitlist_size, uint64_t *waitlist) override;

  virtual int copyBuffer(uint64_t ev_id, uint32_t cq_id, uint32_t src_buffer_id,
                         uint32_t dst_buffer_id,
                         uint32_t content_size_buffer_id, size_t size,
                         size_t src_offset, size_t dst_offset,
                         EventTiming_t &evt, uint32_t waitlist_size,
                         uint64_t *waitlist) override;

  virtual int readBufferRect(uint64_t ev_id, uint32_t cq_id, uint32_t buffer_id,
                             sizet_vec3 &buffer_origin, sizet_vec3 &region,
                             size_t buffer_row_pitch, size_t buffer_slice_pitch,
                             void *host_ptr, size_t host_bytes,
                             EventTiming_t &evt, uint32_t waitlist_size,
                             uint64_t *waitlist) override;

  virtual int writeBufferRect(uint64_t ev_id, uint32_t cq_id,
                              uint32_t buffer_id, sizet_vec3 &buffer_origin,
                              sizet_vec3 &region, size_t buffer_row_pitch,
                              size_t buffer_slice_pitch, void *host_ptr,
                              size_t host_bytes, EventTiming_t &evt,
                              uint32_t waitlist_size,
                              uint64_t *waitlist) override;

  virtual int copyBufferRect(uint64_t ev_id, uint32_t cq_id,
                             uint32_t dst_buffer_id, uint32_t src_buffer_id,
                             sizet_vec3 &dst_origin, sizet_vec3 &src_origin,
                             sizet_vec3 &region, size_t dst_row_pitch,
                             size_t dst_slice_pitch, size_t src_row_pitch,
                             size_t src_slice_pitch, EventTiming_t &evt,
                             uint32_t waitlist_size,
                             uint64_t *waitlist) override;

  virtual int fillBuffer(uint64_t ev_id, uint32_t cq_id, uint32_t buffer_id,
                         size_t offset, size_t size, void *pattern,
                         size_t pattern_size, EventTiming_t &evt,
                         uint32_t waitlist_size, uint64_t *waitlist) override;

  virtual int runKernel(uint64_t ev_id, uint32_t cq_id, uint32_t device_id,
                        uint16_t has_new_args, size_t arg_count, uint64_t *args,
                        unsigned char *is_svm_ptr, size_t pod_size,
                        char *pod_buf, EventTiming_t &evt, uint32_t kernel_id,
                        uint32_t waitlist_size, uint64_t *waitlist,
                        unsigned dim, const sizet_vec3 &offset,
                        const sizet_vec3 &global,
                        const sizet_vec3 *local = nullptr) override;

  /**********************************************************************/
  /**********************************************************************/
  /**********************************************************************/

  virtual int fillImage(uint64_t ev_id, uint32_t cq_id, uint32_t image_id,
                        sizet_vec3 &origin, sizet_vec3 &region,
                        void *fill_color, EventTiming_t &evt,
                        uint32_t waitlist_size, uint64_t *waitlist) override;

  virtual int copyImage2Buffer(uint64_t ev_id, uint32_t cq_id,
                               uint32_t image_id, uint32_t dst_buf_id,
                               sizet_vec3 &origin, sizet_vec3 &region,
                               size_t offset, EventTiming_t &evt,
                               uint32_t waitlist_size,
                               uint64_t *waitlist) override;

  virtual int copyBuffer2Image(uint64_t ev_id, uint32_t cq_id,
                               uint32_t image_id, uint32_t src_buf_id,
                               sizet_vec3 &origin, sizet_vec3 &region,
                               size_t offset, EventTiming_t &evt,
                               uint32_t waitlist_size,
                               uint64_t *waitlist) override;

  virtual int copyImage2Image(uint64_t ev_id, uint32_t cq_id,
                              uint32_t dst_image_id, uint32_t src_image_id,
                              sizet_vec3 &dst_origin, sizet_vec3 &src_origin,
                              sizet_vec3 &region, EventTiming_t &evt,
                              uint32_t waitlist_size,
                              uint64_t *waitlist) override;

  virtual int readImageRect(uint64_t ev_id, uint32_t cq_id, uint32_t image_id,
                            sizet_vec3 &origin, sizet_vec3 &region,
                            void *host_ptr, size_t host_bytes,
                            EventTiming_t &evt, uint32_t waitlist_size,
                            uint64_t *waitlist) override;

  virtual int writeImageRect(uint64_t ev_id, uint32_t cq_id, uint32_t image_id,
                             sizet_vec3 &origin, sizet_vec3 &region,
                             void *host_ptr, size_t host_bytes,
                             EventTiming_t &evt, uint32_t waitlist_size,
                             uint64_t *waitlist) override;

private:
  cl::Buffer *findBuffer(uint32_t id);
  cl::Image *findImage(uint32_t id);
  clKernelStruct *findKernel(uint32_t id);
  cl::Sampler *findSampler(uint32_t id);
  cl::CommandQueue *findCommandQueue(uint32_t id);
  void updateKernelArgMDFromSPIRV(ArgumentInfo_t &MD,
                                  const SPIRVParser::OCLArgTypeInfo &AInfo);
  int createBufferFromSVMRegion(BufferId_t BufferID, size_t Size,
                                cl_mem_flags Flags, void *HostPtr,
                                void **DeviceAddr);
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

/****************************************************************************************************************/

cl::Buffer *SharedCLContext::findBuffer(uint32_t id) {
  auto search = BufferIDmap.find(id);
  return (search == BufferIDmap.end() ? nullptr : search->second.get());
}

cl::Image *SharedCLContext::findImage(uint32_t id) {
  auto search = ImageIDmap.find(id);
  return (search == ImageIDmap.end() ? nullptr : search->second.get());
}

clKernelStruct *SharedCLContext::findKernel(uint32_t id) {
  auto search = KernelIDmap.find(id);
  return (search == KernelIDmap.end() ? nullptr : search->second.get());
}

cl::Sampler *SharedCLContext::findSampler(uint32_t id) {
  auto search = SamplerIDmap.find(id);
  return (search == SamplerIDmap.end() ? nullptr : search->second.get());
}

cl::CommandQueue *SharedCLContext::findCommandQueue(uint32_t id) {
  auto cq_search = QueueIDMap.find(id);
  return (cq_search == QueueIDMap.end() ? nullptr : cq_search->second.get());
}

void SharedCLContext::updateKernelArgMDFromSPIRV(
    ArgumentInfo_t &MD, const SPIRVParser::OCLArgTypeInfo &AInfo) {
  // This is largely a copy-paste from pocl_level0_setup_metadata(),
  // with mainly the destination datatype is changed.

  cl_kernel_arg_address_qualifier Addr;
  cl_kernel_arg_access_qualifier Access;
  Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
  Access = CL_KERNEL_ARG_ACCESS_NONE;
  strncpy(MD.name, AInfo.Name.c_str(), MAX_PACKED_STRING_LEN);
  MD.type_name[0] = 0;

  switch (AInfo.Type) {
  case SPIRVParser::OCLType::POD: {
    MD.type = PoclRemoteArgType::POD;
    break;
  }
  case SPIRVParser::OCLType::Pointer: {
    MD.type = PoclRemoteArgType::Pointer;
    switch (AInfo.Space) {
    case SPIRVParser::OCLSpace::Private:
      Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
      break;
    case SPIRVParser::OCLSpace::Local:
      Addr = CL_KERNEL_ARG_ADDRESS_LOCAL;
      break;
    case SPIRVParser::OCLSpace::Global:
      Addr = CL_KERNEL_ARG_ADDRESS_GLOBAL;
      break;
    case SPIRVParser::OCLSpace::Constant:
      Addr = CL_KERNEL_ARG_ADDRESS_CONSTANT;
      break;
    case SPIRVParser::OCLSpace::Unknown:
      Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
      break;
    }
    break;
  }
  case SPIRVParser::OCLType::Image: {
    MD.type = PoclRemoteArgType::Image;
    Addr = CL_KERNEL_ARG_ADDRESS_GLOBAL;
    bool Readable = AInfo.Attrs.ReadableImg;
    bool Writable = AInfo.Attrs.WriteableImg;
    if (Readable && Writable) {
      Access = CL_KERNEL_ARG_ACCESS_READ_WRITE;
    }
    if (Readable && !Writable) {
      Access = CL_KERNEL_ARG_ACCESS_READ_ONLY;
    }
    if (!Readable && Writable) {
      Access = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
    }
    break;
  }
  case SPIRVParser::OCLType::Sampler: {
    MD.type = PoclRemoteArgType::Sampler;
    break;
  }
  case SPIRVParser::OCLType::Opaque: {
    POCL_MSG_WARN("Unknown SPIR-V argument type 'Opaque', ignoring.\n");
    MD.type = PoclRemoteArgType::POD;
    break;
  }
  }
  MD.address_qualifier = Addr;
  MD.access_qualifier = Access;
  MD.type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
  if (AInfo.Attrs.Constant) {
    MD.type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
  }
  if (AInfo.Attrs.Restrict) {
    MD.type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
  }
  if (AInfo.Attrs.Volatile) {
    MD.type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
  }
}

#define FIND_QUEUE                                                             \
  cq = findCommandQueue(cq_id);                                                \
  if (cq == nullptr) {                                                         \
    POCL_MSG_ERR("CAN'T FIND QUEUE %" PRIu32 " \n", cq_id);                    \
    return CL_INVALID_COMMAND_QUEUE;                                           \
  }

#define FIND_BUFFER                                                            \
  b = findBuffer(buffer_id);                                                   \
  if (b == nullptr) {                                                          \
    POCL_MSG_ERR("CAN'T FIND BUFFER %" PRIu64 " \n", (uint64_t)buffer_id);     \
    return CL_INVALID_MEM_OBJECT;                                              \
  }

#define FIND_BUFFER2(prefix)                                                   \
  prefix = findBuffer(prefix##_buffer_id);                                     \
  if (prefix == nullptr) {                                                     \
    POCL_MSG_ERR("CAN'T FIND BUFFER %" PRIu64 " \n",                           \
                 (uint64_t)prefix##_buffer_id);                                \
    return CL_INVALID_MEM_OBJECT;                                              \
  }

#define FIND_KERNEL                                                            \
  kernel = findKernel(kernel_id);                                              \
  if (kernel == nullptr) {                                                     \
    POCL_MSG_ERR("CAN'T FIND KERNEL %" PRIu32 " for DEV %" PRIu32 " \n",       \
                 kernel_id, device_id);                                        \
    return CL_INVALID_KERNEL;                                                  \
  }                                                                            \
  k = &kernel->perDeviceKernels[device_id]

#define FIND_IMAGE                                                             \
  img = findImage(image_id);                                                   \
  if (img == nullptr) {                                                        \
    POCL_MSG_ERR("CAN't FIND IMAGE %" PRIu32 " \n", image_id);                 \
    return CL_INVALID_MEM_OBJECT;                                              \
  }

#define FIND_IMAGE2(prefix)                                                    \
  prefix = findImage(prefix##_image_id);                                       \
  if (prefix == nullptr) {                                                     \
    POCL_MSG_ERR("CAN't FIND IMAGE %" PRIu32 " \n", prefix##_image_id);        \
    return CL_INVALID_MEM_OBJECT;                                              \
  }

/**
 * Maps the client-side id-based events to local OpenCL platform's events
 * in the given waitlist.
 */
std::vector<cl::Event>
SharedCLContext::remapWaitlist(size_t num_events, uint64_t *ids, uint64_t dep) {
  std::vector<cl::Event> v;
  v.reserve(num_events);

  std::unique_lock<std::mutex> lock(EventmapMutex);
  for (size_t i = 0; i < num_events; ++i) {
    auto e = Eventmap.find(ids[i]);
    if (e != Eventmap.end()) {
      POCL_MSG_PRINT_EVENTS("%" PRIu64 " depends on %s event %" PRIu64 "\n",
                            dep, e->second.native.get() ? "native" : "user",
                            ids[i]);
      v.push_back(e->second.native.get() ? e->second.native : e->second.user);
    } else {
      POCL_MSG_PRINT_EVENTS("Creating placeholder user event for %" PRIu64
                            "'s dependency on %" PRIu64 "\n",
                            dep, ids[i]);
      cl::UserEvent u(ContextWithAllDevices);
      Eventmap.insert({ids[i], {cl::Event(), u}});
      v.push_back(u);
    }
  }

  return v;
}

SharedCLContext::SharedCLContext(cl::Platform *p, unsigned pid,
                                 VirtualContextBase *v,
                                 ReplyQueueThread *s, ReplyQueueThread *f) {
  p->getDevices(CL_DEVICE_TYPE_ALL, &CLDevices);

  cl_context_properties Properties[] = {
      CL_CONTEXT_PLATFORM, reinterpret_cast<intptr_t>(p->operator()()),
      0}; // TODO properties
  ContextWithAllDevices = cl::Context(CLDevices, Properties);

  hasImageSupport = false;
  slow = s;
  fast = f;
  assert(slow);
  assert(fast);
  ParentCtx = v;
  plat_id = pid;

  if (CLDevices.size() == 0) {
    POCL_MSG_ERR("Platform %u has no devices!\n", pid);
    return;
  } else
    POCL_MSG_PRINT_INFO("Platform %u has %" PRIuS " devices\n", pid,
                        CLDevices.size());

  std::string exts = p->getInfo<CL_PLATFORM_EXTENSIONS>();

  for (auto Dev : CLDevices) {
    if (Dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>()) {
      hasImageSupport = true;
      //     POCL_MSG_PRINT_GENERAL("Has image support: {}", hasImageSupport);
      break;
    }
  }

  // create default queues, for memobj migrations
  for (size_t i = 0; i < CLDevices.size(); ++i) {
    QueueIDMap[DEFAULT_QUE_ID + i] = clCommandQueuePtr(new cl::CommandQueue(
        ContextWithAllDevices, CLDevices[i])); // TODO QUEUE_PROPERTIES
    QueueThreadMap[DEFAULT_QUE_ID + i] =
        CommandQueueUPtr(new CommandQueue(this, (DEFAULT_QUE_ID + i), i, s, f));
  }

#if !defined(CLANG) || !defined(LLVM_SPIRV)
  // We require CLANG and LLVM_SPIRV for manipulating the SPIRVs to adjust
  // mismatching client/host SVM pool offsets.
  SVMRegionsStartAddress = nullptr;
  SVMRegionsEndAddress = nullptr;

#warning Disabled PoCL-R SVM due to missing Clang or LLVM-SPIRV
  return;
#else
  if (!pocl_get_bool_option("POCLD_COARSE_GRAIN_SVM", 0)) {
    SVMRegionsStartAddress = nullptr;
    SVMRegionsEndAddress = nullptr;
    return;
  }
#endif

  size_t MaxSVMAllocSize = SIZE_MAX;
  size_t MaxTotalAllocatableSVM = SIZE_MAX;
  for (auto Dev : CLDevices) {
    std::string Extensions = CLDevices[0].getInfo<CL_DEVICE_EXTENSIONS>();
    // We require SPIR-V input to perform the compiler-based SVM offsetting
    // trickery.
    if (Extensions.find("cl_khr_il_program") == std::string::npos)
      continue;
    if ((Dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>() &
        CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) == 0)
      continue;
    CLDevicesWithSVMSupport.push_back(Dev);
    size_t MaxMemAllocSize = Dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    if (MaxMemAllocSize < MaxSVMAllocSize)
      MaxSVMAllocSize = MaxMemAllocSize;

    MaxTotalAllocatableSVM = std::min(MaxTotalAllocatableSVM,
                                      Dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
  }

  if (CLDevicesWithSVMSupport.size() == 0)
    return;

  // Allocate the SVM regions by requesting the largest allocation size the
  // device allows via clSVMAlloc() calls until it starts to return nullptr. It
  // create a set of pinned contiguous regions to map CG SVM allocations to.
  // We'll later communicate the smallest start address and the ending address
  // of the region where the subregions are allocated from, so the client can
  // setup its own virtual SVM region to the host process.

  size_t RemainingAllowedAddrSpaceWaste = SVMMaxAllowedWasteSpace;
  size_t TotalAllocatableSVM = 0;
  do {
    // Do we need a separate context with the SVM devices only? clSVMAlloc()
    // should allocate only from SVM-capable devices, right? The specs is
    // not very clear here. It gets difficult if we have multiple CL contexts
    // at the same time, so we might need to restrict SVM support only to
    // cases where all the remote (at least one a single node) devices support
    // the SVM allocation.
    // ContextWithSVMDevices = cl::Context(CLDevicesWithSVMSupport, Properties);

    if (TotalAllocatableSVM + MaxSVMAllocSize > MaxTotalAllocatableSVM)
      break;

    void *NewSVMRegionAddr = clSVMAlloc(ContextWithAllDevices.get(),
                                        CL_MEM_READ_WRITE, MaxSVMAllocSize, 0);

    if (NewSVMRegionAddr == nullptr)
      // We've likely allocated most of the SVM/global mem. We could try smaller
      // allocations to fill it up, but let's leave it for the future.
      break;

    if (SVMRegionsStartAddress == nullptr) {
      // The first allocated SVM region.
      SVMRegionsStartAddress = NewSVMRegionAddr;
      SVMRegionsEndAddress =
          (void *)((size_t)NewSVMRegionAddr + MaxSVMAllocSize);
    } else {
      // This is not the first allocated SVM region, compute the address space
      // waste.
      size_t AddedAddrSpaceWaste = 0;
      if (NewSVMRegionAddr < SVMRegionsStartAddress) {
        // The new region was allocatead before the previous head.
        AddedAddrSpaceWaste = (size_t)SVMRegionsStartAddress -
                              ((size_t)NewSVMRegionAddr + MaxSVMAllocSize);
      } else if (NewSVMRegionAddr > SVMRegionsEndAddress) {
        // Region after the tail.
        AddedAddrSpaceWaste =
            (size_t)NewSVMRegionAddr - (size_t)SVMRegionsEndAddress;
      } else {
        // Allocation "in the middle" of previous SVM regions, which is great
        // as it reduces the AS waste between the regions it lands to with its
        // size!
        RemainingAllowedAddrSpaceWaste += MaxSVMAllocSize;
      }
      if (AddedAddrSpaceWaste > RemainingAllowedAddrSpaceWaste) {
        POCL_MSG_PRINT_MEMORY(
            "An allocation at %p would have caused too much AS waste.",
            NewSVMRegionAddr);
        clSVMFree(ContextWithAllDevices.get(), NewSVMRegionAddr);
        break;
      }
      RemainingAllowedAddrSpaceWaste -= AddedAddrSpaceWaste;

      SVMRegionsStartAddress =
          std::min(SVMRegionsStartAddress, NewSVMRegionAddr);
      SVMRegionsEndAddress =
          (void *)std::max((size_t)SVMRegionsEndAddress,
                           (size_t)NewSVMRegionAddr + MaxSVMAllocSize);
    }

    SVMRegion NewRegion;
    NewRegion.Allocations = new memory_region;

    pocl_init_mem_region(NewRegion.Allocations,
                         (memory_address_t)NewSVMRegionAddr, MaxSVMAllocSize);
    // Always align to the maximum required alignment of clSVMAlloc().
    NewRegion.Allocations->alignment = 128;
    NewRegion.StartAddress = NewSVMRegionAddr;
    NewRegion.Size = MaxSVMAllocSize;
    cl_int Err;
    NewRegion.ShadowBuffer = clCreateBuffer(
        ContextWithAllDevices.get(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        MaxSVMAllocSize, NewSVMRegionAddr, &Err);
    if (Err != CL_SUCCESS) {
      POCL_MSG_ERR(
          "Unable to allocate a parent SVM cl_mem buffer with size %zu.\n",
          MaxSVMAllocSize);
      clSVMFree(ContextWithAllDevices.get(), NewSVMRegionAddr);
      break;
    } else {
      TotalAllocatableSVM += MaxSVMAllocSize;
      SVMRegions.push_back(NewRegion);
      POCL_MSG_PRINT_MEMORY("PoCL-D allocated an SVM region of size %zu at %p. "
                            "Total allocatable SVM now %zu MB.\n",
                            MaxSVMAllocSize, NewSVMRegionAddr,
                            TotalAllocatableSVM / (1024 * 1024));
    }
  } while (true);
}

SharedCLContext::~SharedCLContext() {
  for (auto &SVMRegion : SVMRegions) {
    POCL_MSG_PRINT_MEMORY("Freeing an SVM region starting at %p.\n",
                          SVMRegion.StartAddress);
    clSVMFree(ContextWithAllDevices.get(), SVMRegion.StartAddress);
    clReleaseMemObject(SVMRegion.ShadowBuffer);
    delete SVMRegion.Allocations;
  }
  SVMRegions.clear();
}

EventPair SharedCLContext::getEventPairForId(uint64_t event_id) {
  std::unique_lock<std::mutex> lock(EventmapMutex);
  auto e = Eventmap.find(event_id);
  if (e != Eventmap.end()) {
    return e->second;
  } else {
    EventPair p{cl::Event(), cl::UserEvent(ContextWithAllDevices)};
    Eventmap.insert({event_id, p});
    return p;
  }
}

int SharedCLContext::waitAndDeleteEvent(uint64_t event_id) {
  std::unique_lock<std::mutex> lock(EventmapMutex);
  auto e = Eventmap.find(event_id);
  if (e != Eventmap.end()) {
    cl::Event ev = e->second.native;
    lock.unlock();
    int r = ev.wait();
    lock.lock();
    auto e = Eventmap.find(event_id);
    if (e != Eventmap.end()) {
      Eventmap.erase(e);
    }
    return r;
  } else {
    // this is used for the fake event in MigrateD2D so don't bother adding user
    // events here
    POCL_MSG_ERR("WaitAndDeledeEvent: no CL event exists for event %" PRIu64
                 "\n",
                 event_id);
    return CL_INVALID_EVENT;
  }
}

/****************************************************************************************************************/
/****************************************************************************************************************/

void SharedCLContext::queuedPush(Request *req) {
  // handle default queues
  if (req->req.cq_id == DEFAULT_QUE_ID) {
    req->req.cq_id += req->req.did;
  }

  if (isCommandReceived(req->req.event_id)) {
    delete req;
    return;
  }

  uint32_t cq_id = req->req.cq_id;
  POCL_MSG_PRINT_GENERAL("SHCTX %u QUEUED PUSH QID %" PRIu32 " DID %" PRIu32
                         "\n",
                         plat_id, cq_id, uint32_t(req->req.did));

  {
    std::unique_lock<std::mutex> lock(MainMutex);
    // TODO reply fail
    assert(QueueIDMap.find(cq_id) != QueueIDMap.end());
    CommandQueue *cq = QueueThreadMap[cq_id].get();
    assert(cq != nullptr);
    cq->push(req);
  }
}

void SharedCLContext::notifyEvent(uint64_t id, cl_int status) {
  std::unique_lock<std::mutex> lock(EventmapMutex);
  auto e = Eventmap.find(id);
  if (e != Eventmap.end()) {
    if (e->second.user.get()) {
      assert(e->second.user.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() >
             CL_COMPLETE);
      e->second.user.setStatus(status);
      POCL_MSG_PRINT_EVENTS("%" PRIu64 ": updating existing user event\n", id);
    } else {
      POCL_MSG_PRINT_EVENTS(
          "%" PRIu64 ": only native event exists, doing nothing\n", id);
    }
  } else {
    cl::UserEvent u(ContextWithAllDevices);
    u.setStatus(status);
    Eventmap.insert({id, {cl::Event(), u}});
    POCL_MSG_PRINT_EVENTS(
        "no event %" PRIu64 " found, creating new user event\n", id);
  }
  for (auto &q : QueueThreadMap) {
    q.second->notify();
  }
}

bool SharedCLContext::isCommandReceived(uint64_t id) {
  std::unique_lock<std::mutex> lock(EventmapMutex);
  auto e = Eventmap.find(id);
  if (e != Eventmap.end()) {
    if (e->second.native.get() ||
        (e->second.user.get() &&
         e->second.user.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() ==
             CL_COMPLETE))
      return true;
  }
  return false;
}

/****************************************************************************************************************/
/****************************************************************************************************************/

#define DI devi.supported_image_formats[i]
static void appendImageFormats(DeviceInfo_t &devi, unsigned i,
                               cl_mem_object_type typ,
                               std::vector<cl::ImageFormat> &formats) {
  if (formats.size() == 0)
    return;

  DI.memobj_type = typ;
  DI.num_formats = formats.size();
  assert(DI.num_formats < MAX_IMAGE_FORMAT_TYPES);
  for (size_t j = 0; j < DI.num_formats; ++j) {
    DI.formats[j].channel_data_type = formats.at(j).image_channel_data_type;
    DI.formats[j].channel_order = formats.at(j).image_channel_order;
  }
}
#undef DI

int SharedCLContext::getDeviceInfo(uint32_t device_id, DeviceInfo_t &i,
                                   std::vector<std::string>& strings) {

  bool is_nvidia = false;
  bool is_pocl_CPU = false;
  POCL_MSG_PRINT_INFO("P %u Get Device %" PRIu32 " Info\n", plat_id, device_id);

  cl::Device clientDevice = CLDevices[device_id];

  std::string temp;

  uint64_t string_offset = 1;
#define PUSH_STRING(ATTR, SRC_STR)                                             \
  do {                                                                         \
    ATTR = string_offset;                                                      \
    strings.push_back(SRC_STR);                                                \
    string_offset += strings.back().size() + 1;                                \
  } while (false)

  PUSH_STRING(i.name, clientDevice.getInfo<CL_DEVICE_NAME>());
  PUSH_STRING(i.opencl_c_version, clientDevice.getInfo<CL_DEVICE_OPENCL_C_VERSION>());
  temp = clientDevice.getInfo<CL_DEVICE_VERSION>();
  PUSH_STRING(i.device_version, temp);
  is_pocl_CPU = (temp.find("pocl") != std::string::npos);

  temp = clientDevice.getInfo<CL_DRIVER_VERSION>();
  PUSH_STRING(i.driver_version, temp);

  temp = clientDevice.getInfo<CL_DEVICE_VENDOR>();
  PUSH_STRING(i.vendor, temp);
  is_nvidia = (temp.find("NVIDIA") != std::string::npos);

  PUSH_STRING(i.builtin_kernels, clientDevice.getInfo<CL_DEVICE_BUILT_IN_KERNELS>());

  // Filter the extensions list and drop those that are currently not
  // supported through PoCL-R.
  std::stringstream extsStream(clientDevice.getInfo<CL_DEVICE_EXTENSIONS>());
  std::string extName;
  std::string exts;

  const std::vector<std::string> unsupportedExts{
      // need to delegate
      // device.getInfo<CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL>()
      "cl_intel_command_queue_families",
      // need to delegate various extended device queries
      "cl_intel_device_attribute_query",
      // need to delegate device.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>()
      "cl_intel_required_subgroup_size",
      // USM/SVM not yet supported.
      "cl_intel_unified_shared_memory",
      // need to delegate device.getInfo<CL_DEVICE_{L,U}UID_KHR>()
      "cl_khr_device_uuid",
      // supports only SPIR-V input (cl_khr_il_program), not the old SPIRs
      "cl_khr_spir",
      // need to delegate
      // device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>() etc.
      "cl_nv_device_attribute_query",
      // pinned buffers are supported only if CG SVM can be enabled
      CL_POCL_PINNED_BUFFERS_EXTENSION_NAME,
  };

  while (getline(extsStream, extName, ' ')) {

    if (std::find(unsupportedExts.begin(), unsupportedExts.end(), extName) !=
        unsupportedExts.end())
      continue;

    if (extName == "cl_khr_il_program") {
      PUSH_STRING(i.supported_spir_v_versions, clientDevice.getInfo<CL_DEVICE_IL_VERSION>());
    }
    if (exts != "")
      exts += " ";
    exts += extName;
  }

  PUSH_STRING(i.extensions, exts);

  i.vendor_id = clientDevice.getInfo<CL_DEVICE_VENDOR_ID>();
  i.address_bits = clientDevice.getInfo<CL_DEVICE_ADDRESS_BITS>();
  i.mem_base_addr_align = clientDevice.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
  // API returns this in bits, but pocl internally uses bytes
  i.mem_base_addr_align /= 8;

  i.global_mem_cache_size =
      clientDevice.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
  i.global_mem_cache_type =
      clientDevice.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>();
  i.global_mem_size = clientDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  i.global_mem_cacheline_size =
      clientDevice.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();

  i.double_fp_config = clientDevice.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>();
  i.single_fp_config = clientDevice.getInfo<CL_DEVICE_SINGLE_FP_CONFIG>();
  i.half_fp_config = clientDevice.getInfo<CL_DEVICE_HALF_FP_CONFIG>();

  i.local_mem_size = clientDevice.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  i.local_mem_type = clientDevice.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();
  i.max_clock_frequency = clientDevice.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  i.max_compute_units = clientDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

  i.max_constant_args = clientDevice.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>();
  i.max_constant_buffer_size =
      clientDevice.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
  i.max_mem_alloc_size = clientDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  i.max_parameter_size = clientDevice.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();

  i.max_read_image_args = clientDevice.getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>();
  i.max_write_image_args =
      clientDevice.getInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>();
  i.max_samplers = clientDevice.getInfo<CL_DEVICE_MAX_SAMPLERS>();

  if (SVMRegions.size() > 0) {
    // For distributed coarse grain SVM.
    i.svm_pool_start_address = (uint64_t)SVMRegionsStartAddress;
    i.svm_pool_size =
        (size_t)SVMRegionsEndAddress - (size_t)SVMRegionsStartAddress;
  } else {
    i.svm_pool_start_address = 0;
    i.svm_pool_size = 0;
  }

  i.max_work_item_dimensions =
      clientDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
  i.max_work_group_size = clientDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  std::vector<size_t> wi =
      clientDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  i.max_work_item_size_x = wi[0];
  i.max_work_item_size_y = wi[1];
  i.max_work_item_size_z = wi[2];

  /* ############  CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE DEPRECATED */

  i.native_vector_width_char =
      clientDevice.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR>();
  i.native_vector_width_short =
      clientDevice.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>();
  i.native_vector_width_int =
      clientDevice.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>();
  i.native_vector_width_long =
      clientDevice.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG>();
  i.native_vector_width_float =
      clientDevice.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>();
  i.native_vector_width_double =
      clientDevice.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
  i.native_vector_width_half =
      clientDevice.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>();

  i.preferred_vector_width_char =
      clientDevice.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>();
  i.preferred_vector_width_short =
      clientDevice.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>();
  i.preferred_vector_width_int =
      clientDevice.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>();
  i.preferred_vector_width_long =
      clientDevice.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>();
  i.preferred_vector_width_float =
      clientDevice.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
  i.preferred_vector_width_double =
      clientDevice.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
  i.preferred_vector_width_half =
      clientDevice.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>();

  /* ############# SUBDEVICES - later */

  // TODO
  // i.printf_buffer_size =
  // clientDevice.getInfo<CL_DEVICE_PRINTF_BUFFER_SIZE>();
  i.printf_buffer_size = 1 << 20;
  i.profiling_timer_resolution =
      clientDevice.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>();

  cl_device_type t = clientDevice.getInfo<CL_DEVICE_TYPE>();
  switch (t) {
  case CL_DEVICE_TYPE_CPU:
    i.type = DevType::CPU;
    break;
  case CL_DEVICE_TYPE_GPU:
    i.type = DevType::GPU;
    break;
  case CL_DEVICE_TYPE_ACCELERATOR:
    i.type = DevType::ACCELERATOR;
    break;
  case CL_DEVICE_TYPE_CUSTOM:
    i.type = DevType::CUSTOM;
    break;
  default:
    assert(false && "unknown dev type");
  }
  i.available = clientDevice.getInfo<CL_DEVICE_AVAILABLE>();
  i.compiler_available = clientDevice.getInfo<CL_DEVICE_COMPILER_AVAILABLE>();
  i.endian_little = clientDevice.getInfo<CL_DEVICE_ENDIAN_LITTLE>();
  i.error_correction_support =
      clientDevice.getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>();
  std::string prof = clientDevice.getInfo<CL_DEVICE_PROFILE>();
  i.full_profile = (prof == "FULL_PROFILE");

  /* ########### images */

  i.image_support = clientDevice.getInfo<CL_DEVICE_IMAGE_SUPPORT>();

  if (i.image_support == CL_FALSE) {
    POCL_MSG_PRINT_GENERAL("P %u Get Device %" PRIu32 " NO IMAGES\n", plat_id,
                           device_id);
    return 0;
  }

  i.image2d_max_height = clientDevice.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
  i.image2d_max_width = clientDevice.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
  i.image3d_max_height = clientDevice.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
  i.image3d_max_width = clientDevice.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
  i.image3d_max_depth = clientDevice.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();

  i.image_max_buffer_size =
      clientDevice.getInfo<CL_DEVICE_IMAGE_MAX_BUFFER_SIZE>();
  i.image_max_array_size =
      clientDevice.getInfo<CL_DEVICE_IMAGE_MAX_ARRAY_SIZE>();

  /*******************************************************************/

  std::vector<cl::ImageFormat> formats{};
  std::vector<cl::Device> temp_vect{clientDevice};
  cl::Context temp_context(temp_vect);

  temp_context.getSupportedImageFormats(CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE1D,
                                        &formats);
  appendImageFormats(i, 0, CL_MEM_OBJECT_IMAGE1D, formats);

  temp_context.getSupportedImageFormats(CL_MEM_READ_ONLY,
                                        CL_MEM_OBJECT_IMAGE1D_ARRAY, &formats);
  appendImageFormats(i, 1, CL_MEM_OBJECT_IMAGE1D_ARRAY, formats);

  temp_context.getSupportedImageFormats(CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D,
                                        &formats);
  appendImageFormats(i, 2, CL_MEM_OBJECT_IMAGE2D, formats);

  temp_context.getSupportedImageFormats(CL_MEM_READ_ONLY,
                                        CL_MEM_OBJECT_IMAGE2D_ARRAY, &formats);
  appendImageFormats(i, 3, CL_MEM_OBJECT_IMAGE2D_ARRAY, formats);

  temp_context.getSupportedImageFormats(CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE3D,
                                        &formats);
  appendImageFormats(i, 4, CL_MEM_OBJECT_IMAGE3D, formats);

  return 0;
}

/****************************************************************************************************************/
/****************************************************************************************************************/

#ifdef QUEUE_PROFILING
#define QUEUE_PROPERTIES cl::QueueProperties::Profiling
#else
#define QUEUE_PROPERTIES 0
#endif

int SharedCLContext::createQueue(uint32_t queue_id, uint32_t dev_id) {

  cl_int err;
  clCommandQueuePtr p(new cl::CommandQueue(
      ContextWithAllDevices, CLDevices[dev_id], QUEUE_PROPERTIES, &err));
  if (err != CL_SUCCESS) {
    POCL_MSG_ERR("P %u Create Queue\n", plat_id);
    return err;
  }

  CommandQueueUPtr que(new CommandQueue(this, queue_id, dev_id, slow, fast));

  {
    std::unique_lock<std::mutex> lock(MainMutex);
    QueueIDMap[queue_id] = p;
    QueueThreadMap[queue_id] = std::move(que);
  }
  POCL_MSG_PRINT_INFO("P %u Create Queue %" PRIu32 "\n", plat_id, queue_id);
  return 0;
}

int SharedCLContext::freeQueue(uint32_t queue_id) {
  {
    std::unique_lock<std::mutex> lock(MainMutex);
    if (QueueThreadMap.find(queue_id) == QueueThreadMap.end()) {
      POCL_MSG_ERR("P %u Free Queue %" PRIu32 "\n", plat_id, queue_id);
      return CL_INVALID_COMMAND_QUEUE;
    }
    QueueThreadMap.erase(queue_id);
    QueueIDMap.erase(queue_id);
  }
  POCL_MSG_PRINT_INFO("P %u Free Queue %" PRIu32 "\n", plat_id, queue_id);
  return 0;
}

/****************************************************************************************************************/
/****************************************************************************************************************/

#define WRITE_BYTES(var)                                                       \
  std::memcpy(buf, &var, sizeof(var));                                         \
  buf += sizeof(var);                                                          \
  assert((size_t)(buf - buffer) <= buffer_size);
#define WRITE_STRING(str, len)                                                 \
  std::memcpy(buf, str, len);                                                  \
  buf += len;                                                                  \
  assert((size_t)(buf - buffer) <= buffer_size);

#if defined(CLANG) && defined(LLVM_SPIRV)
/**
 * Creates a SPIRV with all global memory addresses adjusted by adding
 * the SVMOffset.
 *
 * Used to adjust to an SVM region offset difference between the device and the
 * host. SVMOffset is asumed to wrap around the addition at unsigned 64b for
 * negative offsets.
 *
 * \param InputSPV The original SPIR-V to manipulate (non-empty if compiling
 * from SPIR-V) \param Src The OpenCL C source code to compile from (non-null if
 * compiling from src) \param SrcSize The size of the OpenCL C source (greater
 * than 0 when compiling from src) \param SVMOffset The SVM offset to add to all
 * global memory addresses. \param NewSPV Vector to push the produced SPIR-V to.
 * \param BuildOptions For the source compilation.
 * \return True if successful, false (and NewSPV empty) if fails.
 */
bool createSPIRVWithSVMOffset(const std::vector<unsigned char> *InputSPV,
                              char *Src, size_t SrcSize, size_t SVMOffset,
                              std::vector<char> &NewSPV,
                              const char *BuildOptions) {

  // Just invoke command line tools for now.
  constexpr int CmdOutputMaxSize = 10000;
  char CmdOutput[CmdOutputMaxSize];
  size_t CmdOutputSize = CmdOutputMaxSize;

  char *TempDirName = strdup(
      (std::filesystem::temp_directory_path() / "pocl-r-XXXXXX").c_str());

  mkdtemp(TempDirName);

  std::filesystem::path TempDir(TempDirName);

  free(TempDirName);

  const std::string OrigBcFileName = TempDir / "original.bc";

  if (Src != nullptr && SrcSize > 0) {
    // Compile from sources.
    const std::string SrcFileName = TempDir / "input.cl";
    std::ofstream SrcFile(SrcFileName);
    SrcFile.write(Src, SrcSize);
    SrcFile.close();

    // https://www.khronos.org/blog/offline-compilation-of-opencl-kernels-into-
    // spir-v-using-open-source-tooling
    std::stringstream OpenCLCCmd;
    OpenCLCCmd << CLANG
               << " -c -target spir64 -cl-kernel-arg-info -cl-std=CL3.0 "
               << SrcFileName.c_str() << " " << BuildOptions
               << " -emit-llvm -o " << OrigBcFileName.c_str();

    if (system(OpenCLCCmd.str().c_str()) != EXIT_SUCCESS)
      return false;

  } else if (InputSPV != nullptr) {
    const std::string OrigSpvFileName = TempDir / "original.spv";

    std::ofstream OrigSpvFile(OrigSpvFileName);
    OrigSpvFile.write((const char *)InputSPV->data(), InputSPV->size());
    OrigSpvFile.close();

    std::stringstream SpvCmd;

    SpvCmd << LLVM_SPIRV << " -r " << OrigSpvFileName.c_str() << " -o "
           << OrigBcFileName.c_str();

    if (system(SpvCmd.str().c_str()) != EXIT_SUCCESS)
      return false;

  } else {
    assert(false && "Unimplemented.");
  }

  const std::string OffsettedBcFileName = TempDir / "offsetted.bc";

  std::filesystem::path LibPoCLPath;
  std::stringstream OptCmd;

  if (!pocl_get_bool_option("POCL_BUILDING", 0))
    LibPoCLPath /= std::filesystem::path(POCL_INSTALL_LIBDIR) / "libpocl.so";
  else
    LibPoCLPath /=
        std::filesystem::path(BUILDDIR) / "lib" / "CL" / "libpocl.so";

  OptCmd << LLVM_OPT << " -load-pass-plugin=" << LibPoCLPath
         << " -passes=svm-offset -svm-offset-value=" << SVMOffset << " "
         << OrigBcFileName << " -o " << OffsettedBcFileName;

  if (system(OptCmd.str().c_str()) != EXIT_SUCCESS)
    return false;

  const std::string OutSpvFileName = TempDir / "offsetted.spv";

  std::stringstream SpvCmd;

  SpvCmd << LLVM_SPIRV << " " << OffsettedBcFileName.c_str() << " -o "
         << OutSpvFileName.c_str();

  if (system(SpvCmd.str().c_str()) != EXIT_SUCCESS)
    return false;

  std::ifstream OutFile(OutSpvFileName);
  char C;
  NewSPV.clear();
  while (OutFile.read((char *)&C, 1)) {
    NewSPV.push_back(C);
  }

  // std::filesystem::remove_all(TempDir);
  return true;
}
#endif

int SharedCLContext::buildOrLinkProgram(
    uint32_t program_id, std::vector<uint32_t> &DeviceList, char *src,
    size_t src_size, bool is_binary, bool is_builtin, bool is_spirv,
    const char *options,
    std::unordered_map<uint64_t, std::vector<unsigned char>> &InputBinaries,
    std::unordered_map<uint64_t, std::vector<unsigned char>> &output_binaries,
    std::unordered_map<uint64_t, std::string> &build_logs, size_t &num_kernels,
    uint64_t SVMRegionOffset, bool CompileOnly, bool LinkOnly) {

  cl_int err = 0;
  cl::Program *p = nullptr;
  assert(ProgramIDmap.find(program_id) == ProgramIDmap.end());
  clProgramStructPtr program_uptr(new clProgramStruct{});
  clProgramStruct *program = program_uptr.get();
  std::vector<cl::Kernel> prebuilt_kernels;
  SPIRVParser::OpenCLFunctionInfoMap KernelInfoMap;

  bool AlwaysBuildAll = DeviceList.empty();
  for (auto i : DeviceList) {
    std::string vendor = CLDevices[i].getInfo<CL_DEVICE_VENDOR>();
    std::string device_version = CLDevices[i].getInfo<CL_DEVICE_VERSION>();
    if (vendor.find("NVIDIA") != std::string::npos &&
        !(device_version.find("PoCL") != std::string::npos &&
          device_version.find("CUDA") != std::string::npos))
      AlwaysBuildAll = true;
  }

  POCL_MSG_PRINT_INFO("P %u Building Program %" PRIu32 "\n", plat_id,
                      program_id);

  program->devices.resize(DeviceList.size());
  assert(DeviceList.size() > 0);
  for (size_t i = 0; i < DeviceList.size(); ++i) {
    assert(DeviceList[i] < CLDevices.size());
    program->devices[i] = CLDevices[DeviceList[i]];
  }

  if (options == nullptr)
    options = "";

  std::string opts(options);

  /* Kernel argument information is only available when building
     from sources, but some implementations seem to return metadata
     also for binaries/SPIR-V.

     https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/
     clGetKernelArgInfo.html*/
  opts += " -cl-kernel-arg-info";

  if (LinkOnly) {
    // Collect the previously built programs from the server-side cache and link
    // them.
    std::vector<cl_program> InputPrograms;
    for (auto &E : InputBinaries) {
      uint32_t ClientProgramID = E.first;
      if (ProgramIDmap.find(ClientProgramID) == ProgramIDmap.end()) {
        POCL_MSG_ERR("Unable to find program with id %u in the ID map.",
                     ClientProgramID);
      }
      InputPrograms.push_back(ProgramIDmap[ClientProgramID]->uptr->get());
    }

    cl_program LinkedProgram = ::clLinkProgram(
        ContextWithAllDevices.get(), 0, nullptr, options,
        static_cast<cl_uint>(InputPrograms.size()),
        reinterpret_cast<const cl_program *>(InputPrograms.data()), nullptr,
        nullptr, &err);

    if (err != CL_SUCCESS) {
      POCL_MSG_ERR("clLinkProgram() failed\n");
      return err;
    }

    clProgramPtr Prog(new cl::Program(LinkedProgram));

    p = Prog.get();
    program->uptr = std::move(Prog);

  } else if (SVMRegionOffset > 0 && !is_builtin && !is_binary) {
    std::vector<char> SVMOffsettedSPIRV;


#if defined(CLANG) && defined(LLVM_SPIRV)
    // Adjust the SVM region offset to the kernel code.
    bool SuccessfulOffsetting = createSPIRVWithSVMOffset(
        is_spirv ? &(*InputBinaries.begin()).second : nullptr, src, src_size,
        SVMRegionOffset, SVMOffsettedSPIRV, options);
#else
    bool SuccessfulOffsetting = false;
#endif

    assert(SuccessfulOffsetting && SVMOffsettedSPIRV.size() > 0);

    cl_int Err = CL_SUCCESS;
    clProgramPtr Prog(
        new cl::Program(ContextWithAllDevices, SVMOffsettedSPIRV, false, &Err));
    if (Err != CL_SUCCESS) {
      POCL_MSG_ERR("clCreateProgramWithIL() of the offsetted SPIR-V failed\n");
      return Err;
    }

    if (!SPIRVParser::parseSPIRV((const int32_t *)SVMOffsettedSPIRV.data(),
                                 SVMOffsettedSPIRV.size() / 4, KernelInfoMap)) {
      POCL_MSG_ERR("Unable to parse the SVM adjusted SPIR-V for metadata. "
                   "Illegal SPIR-V?\n");
      return CL_INVALID_PROGRAM;
    }

    p = Prog.get();
    program->uptr = std::move(Prog);

    is_spirv = true;

  } else if (is_builtin) {

    std::string source(src, src + src_size);
    {
      POCL_MSG_PRINT_GENERAL("BUILDING BUILTIN KERNELS WITH OPTIONS : %s\n",
                             opts.c_str());

      clProgramPtr pp(new cl::Program(ContextWithAllDevices, program->devices,
                                      source, &err));

      if (err != CL_SUCCESS) {
        POCL_MSG_ERR("CreateProgramWithBuiltinKernels() failed\n");
        return err;
      }

      p = pp.get();
      program->uptr = std::move(pp);
    }

  } else if (is_binary) {
    POCL_MSG_PRINT_GENERAL("BUILDING BINARY WITH OPTIONS : %s\n", opts.c_str());

    cl::Program::Binaries plat_binaries;
    plat_binaries.resize(DeviceList.size());
    for (size_t i = 0; i < DeviceList.size(); ++i) {
      uint64_t id = ((uint64_t)plat_id << 32) + DeviceList[i];
      assert(InputBinaries.find(id) != InputBinaries.end());
      plat_binaries[i] = InputBinaries[id];
    }

    clProgramPtr pp(new cl::Program(ContextWithAllDevices, program->devices,
                                    plat_binaries, nullptr, &err));
    if (err != CL_SUCCESS) {
      POCL_MSG_ERR("CreateProgramWithBinary() failed\n");
      return err;
    }

    p = pp.get();
    program->uptr = std::move(pp);

  } else if (is_spirv) {
    POCL_MSG_PRINT_GENERAL("BUILDING SPIR-V WITH OPTIONS : %s\n", opts.c_str());

    cl::Program::Binaries PlatBinaries;
    PlatBinaries.resize(DeviceList.size());
    for (size_t i = 0; i < DeviceList.size(); ++i) {
      uint64_t id = ((uint64_t)plat_id << 32) + DeviceList[i];
      assert(InputBinaries.find(id) != InputBinaries.end());
      PlatBinaries[i] = InputBinaries[id];
    }

    // Expecting to see a single SPIR-V which is built for all capable
    // devices. Strictly put, we should check the SPIR-Vs are the same
    // for all.
    assert(PlatBinaries.size() >= 1);

    // Annoyingly cl::Program constructor expects 'char' whereas Binaries
    // come with an 'unsigned char' element type. Perhaps just copy it here
    // to avoid problems.
    const std::vector<char> &IL = reinterpret_cast<const std::vector<char> &>(
        (*InputBinaries.begin()).second);

    clProgramPtr pp(new cl::Program(ContextWithAllDevices, IL, false, &err));
    if (err != CL_SUCCESS) {
      POCL_MSG_ERR("clCreateProgramWithIL() failed\n");
      return err;
    }

    // The SPIR-V parser inputs a stream of int32_t. Do we need to
    // realign the blob or can we assume it's aligned when reading
    // in?
    assert(((size_t)IL.data()) % 4 == 0);
    if (!SPIRVParser::parseSPIRV((const int32_t *)IL.data(), IL.size() / 4,
                                 KernelInfoMap)) {
      POCL_MSG_WARN("Unable to parse the SPIR-V for metadata. "
                    "Illegal SPIR-V?\n");
      return CL_INVALID_PROGRAM;
    }

    p = pp.get();
    program->uptr = std::move(pp);
  } else {
    POCL_MSG_PRINT_GENERAL("BUILDING SRC WITH OPTIONS : %s\n", opts.c_str());

    std::string source(src, src + src_size);

    clProgramPtr pp(
        new cl::Program(ContextWithAllDevices, source, false, &err));

    if (err != CL_SUCCESS) {
      POCL_MSG_ERR("CreateProgramWithSource() failed\n");
      return err;
    }

    p = pp.get();
    program->uptr = std::move(pp);
  }

  if (!LinkOnly) {
    // build
    if (AlwaysBuildAll) {
      // XXX: hacky workaround for wonky behaviour with certain drivers
      // when compiling a program for only a subset of the context's devices
      err = CompileOnly ? p->compile(opts.c_str()) : p->build(opts.c_str());
    } else {
      if (CompileOnly) {
        // cl2.hpp doesn't have device-limiting versions of compile()
        // reported in https://github.com/KhronosGroup/OpenCL-CLHPP/issues/285

        std::size_t NumDevices = program->devices.size();
        std::vector<cl_device_id> DeviceIDs(NumDevices);

        for (std::size_t DeviceIndex = 0; DeviceIndex < NumDevices;
             ++DeviceIndex) {
          DeviceIDs[DeviceIndex] = (program->devices[DeviceIndex])();
        }

        err = ::clCompileProgram(p->get(), NumDevices, DeviceIDs.data(),
                                 opts.c_str(), 0, nullptr, nullptr, nullptr,
                                 nullptr);
      } else {
        err = p->build(program->devices, opts.c_str());
      }
    }
  }

  // even if build failed, return build log
  auto buildInfo = p->getBuildInfo<CL_PROGRAM_BUILD_LOG>();
  if (buildInfo.size() > 0) {
    size_t i = 0;
    for (const auto &pair : buildInfo) {
      if (i < DeviceList.size()) {
        // assert (pair.first() == program->devices[i]);
        uint64_t id = ((uint64_t)plat_id << 32) + DeviceList[i];
        std::string buildlog = pair.second;
        build_logs[id] = std::move(buildlog);
        POCL_MSG_PRINT_GENERAL("Platform %u Device %" PRIu32 " Build log: \n%s",
                               plat_id, DeviceList[i], build_logs[id].c_str());
      } else {
        POCL_MSG_PRINT_GENERAL("Platform %u Unknown Device %" PRIuS
                               " Build log: \n%s",
                               plat_id, i, pair.second.c_str());
      }
      ++i;
    }
  }

  if (err != CL_SUCCESS) {
    POCL_MSG_ERR("cl{Link|Build|Compile}Program() FAILED \n");
    return err;
  }

  err = p->createKernels(&prebuilt_kernels);
  if (err) {
    POCL_MSG_ERR("clCreateKernels failed\n");
    return err;
  }

  // for sources, return also the binary
  if (!is_binary && !is_builtin) {
    cl::Program::Binaries binaries;
    assert(binaries.size() == 0);

    /* we could use a much simpler way here: just getInfo(CL_PROGRAM_BINARIES)
       directly and assume the returned array is exactly the same size as
       DeviceList (= devices for which program). This is what the spec says
       should happen. Unfortunately some broken platforms (ARM Mali) return
       vector of binaries larger than the number of device list given as
       argument to clBuildProgram. This happens when 1) you have a context with
       2 devices, 2) you create and build a program for only the 2nd device in
       context
       ... the returned "binaries" vector then has size 2. */

    std::vector<cl::Device> builtProgramDevices =
        p->getInfo<CL_PROGRAM_DEVICES>(&err);
    assert(err == CL_SUCCESS);

    err = p->getInfo<>(CL_PROGRAM_BINARIES, &binaries);
    assert(err == CL_SUCCESS);

    POCL_MSG_PRINT_GENERAL(
        "BPD: %" PRIuS "    PD: %" PRIuS "    BIN: %" PRIuS " \n",
        builtProgramDevices.size(), program->devices.size(), binaries.size());

    assert(builtProgramDevices.size() == binaries.size());

    size_t i, j;
    if (binaries.size() == DeviceList.size()) {
      for (i = 0; i < DeviceList.size(); ++i) {
        uint64_t id = ((uint64_t)plat_id << 32) + DeviceList[i];
        POCL_MSG_PRINT_GENERAL("Writing binary for Dev ID: %u / %" PRIu32 " \n",
                               plat_id, DeviceList[i]);
        output_binaries[id] = std::move(binaries[i]);
      }
      assert(binaries.size() == DeviceList.size());
    } else {
      for (i = 0; i < DeviceList.size(); ++i) {
        cl_device_id dev = program->devices[i].get();
        for (j = 0; j < builtProgramDevices.size(); ++j) {
          cl_device_id match = builtProgramDevices[j].get();
          if (dev == match) {
            size_t orig =
                program->devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            size_t found =
                builtProgramDevices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            POCL_MSG_PRINT_GENERAL("MATCH %" PRIuS " AT INDEX %" PRIuS
                                   "  ||||||| %" PRIuS " / %" PRIuS " \n",
                                   i, j, orig, found);
            break;
          }
        }
        assert(j < builtProgramDevices.size());
        builtProgramDevices.erase(builtProgramDevices.begin() + j);
        POCL_MSG_PRINT_GENERAL("BPD SIZE %" PRIuS "\n",
                               builtProgramDevices.size());

        uint64_t id = ((uint64_t)plat_id << 32) + DeviceList[i];
        POCL_MSG_PRINT_GENERAL("Writing binary for Dev ID: %u / %" PRIu32 " \n",
                               plat_id, DeviceList[i]);
        output_binaries[id] = std::move(binaries[j]);
      }
    }
  }

  // set up kernels
  std::vector<cl::Kernel> &kernels = prebuilt_kernels;
  // a sanity check to ensure we build the same amount of kernels for each
  // device
  if (num_kernels == 0)
    num_kernels = kernels.size();
  else
    assert(num_kernels == kernels.size());

  // Kernel metadata is guaranteed to be available only when building
  // from sources. Let's accumulate what we get from the clGetKernelInfo etc.
  // queries and augment with SPIR-V parser data, if building from SPIR-V.
  program->numKernels = num_kernels;
  program->kernel_meta.resize(num_kernels);
  for (size_t i = 0; i < num_kernels; ++i) {
    KernelMetaInfo_t &temp_kernel = program->kernel_meta[i].meta;
    cl_int ArgErr = CL_SUCCESS;

    temp_kernel.total_local_size = 0;
    temp_kernel.reqd_wg_size = {0, 0, 0};

    std::string kernel_name =
        kernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>(&ArgErr);
    // Assume we get the name always.
    assert(ArgErr == CL_SUCCESS);
    std::strncpy(temp_kernel.name, kernel_name.c_str(), MAX_PACKED_STRING_LEN);

    std::string a = kernels[i].getInfo<CL_KERNEL_ATTRIBUTES>(&ArgErr);
    if (ArgErr == CL_SUCCESS) {
      std::strncpy(temp_kernel.attributes, a.c_str(), MAX_PACKED_STRING_LEN);
    }

    size_t num_args_temp = kernels[i].getInfo<CL_KERNEL_NUM_ARGS>(&ArgErr);
    if (ArgErr == CL_SUCCESS) {
      temp_kernel.num_args = num_args_temp;
      program->kernel_meta[i].arg_meta.resize(temp_kernel.num_args);
    } else if (is_spirv) {
      temp_kernel.num_args = KernelInfoMap[kernel_name]->ArgTypeInfo.size();
    } else {
      temp_kernel.num_args = 0;
    }

    for (cl_uint arg_index = 0; arg_index < temp_kernel.num_args; ++arg_index) {
      ArgumentInfo_t &temp_arg = program->kernel_meta[i].arg_meta[arg_index];

      if (is_spirv) {
        updateKernelArgMDFromSPIRV(temp_arg,
                                   KernelInfoMap[kernel_name]->ArgTypeInfo[arg_index]);
        continue;
      }

      temp_arg.access_qualifier =
          kernels[i].getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index,
                                                                &ArgErr);
      temp_arg.address_qualifier =
          kernels[i].getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index,
                                                                 &ArgErr);
      temp_arg.type_qualifier =
          kernels[i].getArgInfo<CL_KERNEL_ARG_TYPE_QUALIFIER>(arg_index,
                                                              &ArgErr);

      std::string arg_typename =
          kernels[i].getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index, &ArgErr);
      if (ArgErr == CL_SUCCESS) {
        std::strncpy(temp_arg.type_name, arg_typename.c_str(),
                     MAX_PACKED_STRING_LEN);
      }

      std::string arg_name =
          kernels[i].getArgInfo<CL_KERNEL_ARG_NAME>(arg_index, &ArgErr);
      if (ArgErr == CL_SUCCESS) {
        std::strncpy(temp_arg.name, arg_name.c_str(), MAX_PACKED_STRING_LEN);
      }
      // TODO this is hackish, but what else can we do here
      temp_arg.type = PoclRemoteArgType::POD;

      if (temp_arg.access_qualifier != CL_KERNEL_ARG_ACCESS_NONE)
        temp_arg.type = PoclRemoteArgType::Image;

      if (arg_typename.find("sampler_t") != std::string::npos)
        temp_arg.type = PoclRemoteArgType::Sampler;

      if ((temp_arg.address_qualifier != CL_KERNEL_ARG_ADDRESS_PRIVATE) &&
          (arg_typename.back() == '*')) {
        temp_arg.type = PoclRemoteArgType::Pointer;
      }

      POCL_MSG_PRINT_GENERAL(
          "BUILD / KERNEL %s ARG %s / %u / %s : DETERMINED TYPE %d \n",
          kernel_name.c_str(), arg_name.c_str(), arg_index,
          arg_typename.c_str(), PoclRemoteArgType(temp_arg.type));
    }
  }

  if (err)
    return err;

  // SUCCESS
  {
    std::unique_lock<std::mutex> lock(MainMutex);
    ProgramIDmap[program_id] = std::move(program_uptr);
  }

  POCL_MSG_PRINT_INFO("Created & built program %" PRIu32 "\n", program_id);
  return CL_SUCCESS;
}

int SharedCLContext::freeProgram(uint32_t program_id) {
  {
    std::unique_lock<std::mutex> lock(MainMutex);
    if (ProgramIDmap.erase(program_id) == 0) {
      POCL_MSG_ERR("P %u Free Program %" PRIu32 "\n", plat_id, program_id);
      return CL_INVALID_PROGRAM;
    }
  }
  POCL_MSG_PRINT_INFO("P %u Free Program %" PRIu32 "\n", plat_id, program_id);
  return 0;
}

int SharedCLContext::writeKernelMeta(uint32_t program_id, char *buffer,
                                     size_t *written) {
  clProgramStruct *p = nullptr;
  char *buf = buffer;
  size_t buffer_size = MAX_REMOTE_BUILDPROGRAM_SIZE;
  {
    std::unique_lock<std::mutex> lock(MainMutex);
    auto search = ProgramIDmap.find(program_id);
    //  POCL_MSG_ERR ("write kernel meta {}\n", program_id);
    assert(search != ProgramIDmap.end());
    p = search->second.get();
  }

  assert(p);
  // there could be 0 kernels in a program
  // assert (p->kernel_meta.size() > 0);
  std::vector<clKernelMetadata> &meta = p->kernel_meta;
  uint32_t num_kernels = meta.size();
  uint64_t placeholder = 0;

  WRITE_BYTES(placeholder);
  WRITE_BYTES(num_kernels);
  for (size_t i = 0; i < num_kernels; ++i) {
    WRITE_STRING(&meta[i].meta, sizeof(KernelMetaInfo_t));
    uint32_t num_args = meta[i].arg_meta.size();

    WRITE_BYTES(num_args);
    for (size_t j = 0; j < num_args; ++j) {
      WRITE_STRING(&meta[i].arg_meta[j], sizeof(ArgumentInfo_t));
    }
  }

  *written = (size_t)(buf - buffer);
  assert(*written > 0);
  *((uint64_t *)buffer) = (uint64_t)(*written) - sizeof(placeholder);
  return 0;
}

/****************************************************************************************************************/

int SharedCLContext::createKernel(uint32_t kernel_id, uint32_t program_id,
                                  const char *name) {
  POCL_MSG_PRINT_INFO("P %u Create Kernel %" PRIu32 " / %s in program %" PRIu32
                      "\n",
                      plat_id, kernel_id, name, program_id);
  assert(KernelIDmap.find(kernel_id) == KernelIDmap.end());

  clProgramStruct *program = nullptr;
  cl::Program *p = nullptr;
  clKernelStructPtr kernel(new clKernelStruct{});
  clKernelStruct *k = kernel.get();
  std::string namestr(name);

  {
    std::unique_lock<std::mutex> lock(MainMutex);
    if (ProgramIDmap.find(program_id) == ProgramIDmap.end()) {
      POCL_MSG_ERR("P %u Can't find program %" PRIu32 "\n", plat_id,
                   program_id);
      return CL_INVALID_PROGRAM;
    }
    program = ProgramIDmap[program_id].get();
    p = program->uptr.get();
  }

  assert(program);
  assert(p);

  bool found = false;
  for (size_t i = 0; i < program->numKernels; ++i) {
    std::string temp(program->kernel_meta[i].meta.name);
    if (temp == namestr) {
      found = true;
      k->metaData = &program->kernel_meta[i];
      break;
    }
  }

  if (!found) {
    POCL_MSG_ERR("Invalid kernel name: %s\n", name);
    return CL_INVALID_ARG_VALUE;
  }

  k->isFakeBuiltin = program->isFakeBuiltin;
  k->numArgs = k->metaData->meta.num_args;

  // create a separate kernel for each device
  // this is because argument setting needs to be separate for each device
  k->perDeviceKernels.resize(CLDevices.size());
  for (size_t ii = 0; ii < CLDevices.size(); ++ii) {
    k->perDeviceKernels[ii] = cl::Kernel(*p, name);
  }

  {
    std::unique_lock<std::mutex> lock(MainMutex);
    KernelIDmap[kernel_id] = std::move(kernel);
  }

  return CL_SUCCESS;
}

int SharedCLContext::freeKernel(uint32_t kernel_id) {
  {
    std::unique_lock<std::mutex> lock(MainMutex);
    if (KernelIDmap.erase(kernel_id) == 0) {
      POCL_MSG_ERR("P %u Free Kernel %" PRIu32 "\n", plat_id, kernel_id);
      return CL_INVALID_KERNEL;
    }
  }
  POCL_MSG_PRINT_INFO("P %u Free Kernel %" PRIu32 "\n", plat_id, kernel_id);
  return 0;
}

int SharedCLContext::createSampler(uint32_t sampler_id, uint32_t normalized,
                                   uint32_t address, uint32_t filter) {
  CHECK_IMAGE_SUPPORT();
  int err = CL_SUCCESS;
  clSamplerPtr sam(new cl::Sampler(ContextWithAllDevices, normalized, address,
                                   filter, &err));
  if (err != CL_SUCCESS) {
    POCL_MSG_ERR("P %u Create Sampler %" PRIu32 "\n", plat_id, sampler_id);
    return err;
  }

  {
    std::unique_lock<std::mutex> lock(MainMutex);
    SamplerIDmap[sampler_id] = std::move(sam);
  }
  POCL_MSG_PRINT_INFO("P %u Create Sampler %" PRIu32 "\n", plat_id, sampler_id);
  return 0;
}

int SharedCLContext::freeSampler(uint32_t sampler_id) {
  {
    std::unique_lock<std::mutex> lock(MainMutex);
    if (SamplerIDmap.erase(sampler_id) == 0) {
      POCL_MSG_ERR("P %u Free Sampler %" PRIu32 "\n", plat_id, sampler_id);
      return CL_INVALID_MEM_OBJECT;
    }
  }
  CHECK_IMAGE_SUPPORT();
  POCL_MSG_PRINT_INFO("P %u Free Sampler %" PRIu32 "\n", plat_id, sampler_id);
  return 0;
}

int SharedCLContext::createImage(uint32_t image_id, uint32_t flags,
                                 // format
                                 uint32_t channel_order,
                                 uint32_t channel_data_type,
                                 // desc
                                 uint32_t type, uint32_t width, uint32_t height,
                                 uint32_t depth, uint32_t array_size,
                                 uint32_t row_pitch, uint32_t slice_pitch) {

  CHECK_IMAGE_SUPPORT();
  flags = flags & (cl_bitfield)(~(CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR));

  POCL_MSG_PRINT_GENERAL("P %u Create Image || order %" PRIu32 " || "
                         "dtype %" PRIu32 " || type %" PRIu32 " || "
                         "w %" PRIu32 " || h %" PRIu32 " || d %" PRIu32 " || "
                         "A %" PRIu32 " || RP %" PRIu32 " || SP %" PRIu32 " \n",
                         plat_id, channel_order, channel_data_type, type, width,
                         height, depth, array_size, row_pitch, slice_pitch);

  cl::ImageFormat img_format(channel_order, channel_data_type);
  cl_int err = CL_SUCCESS;
  clImagePtr img;
  cl_mem_object_type t = type;
  switch (t) {
  case CL_MEM_OBJECT_IMAGE1D:
    img = clImagePtr(new cl::Image1D(ContextWithAllDevices, flags, img_format,
                                     width, nullptr, &err));
    break;
  case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    img = clImagePtr(new cl::Image1DArray(ContextWithAllDevices, flags,
                                          img_format, array_size, width,
                                          row_pitch, nullptr, &err));
    break;
    //      case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    //        img = clImagePtr(new cl::Image1DBuffer(ContextWithAllDevices,
    //        flags, img_format, width, nullptr, &err)); break;
  case CL_MEM_OBJECT_IMAGE2D:
    img = clImagePtr(new cl::Image2D(ContextWithAllDevices, flags, img_format,
                                     width, height, 0, nullptr, &err));
    break;
  case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    img = clImagePtr(new cl::Image2DArray(
        ContextWithAllDevices, flags, img_format, array_size, width, height,
        row_pitch, slice_pitch, nullptr, &err));
    break;
  case CL_MEM_OBJECT_IMAGE3D:
    img = clImagePtr(new cl::Image3D(ContextWithAllDevices, flags, img_format,
                                     width, height, depth, row_pitch,
                                     slice_pitch, nullptr, &err));
    break;
  default: {
    POCL_MSG_ERR("Create Image: invalid image type %u\n", t);
    return CL_INVALID_IMAGE_DESCRIPTOR;
  }
  }

  if (err != CL_SUCCESS) {
    POCL_MSG_ERR("P %u Create Image %" PRIu32 "\n", plat_id, image_id);
    return err;
  }

  {
    std::unique_lock<std::mutex> lock(MainMutex);
    ImageIDmap[image_id] = std::move(img);
  }
  POCL_MSG_PRINT_INFO("P %u Create Image %" PRIu32 "\n", plat_id, image_id);
  return 0;
}

int SharedCLContext::freeImage(uint32_t image_id) {
  CHECK_IMAGE_SUPPORT();
  {
    std::unique_lock<std::mutex> lock(MainMutex);
    if (ImageIDmap.erase(image_id) == 0) {
      POCL_MSG_ERR("P %u Free Image %" PRIu32 "\n", plat_id, image_id);
      return CL_INVALID_MEM_OBJECT;
    }
  }
  POCL_MSG_PRINT_INFO("P %u Free Image %" PRIu32 "\n", plat_id, image_id);
  return 0;
}

/**
 * Creates a buffer from the preallocated SVM region.
 *
 * When SVM is enabled, both types of buffers (SVM and cl_mem) are allocated
 * from the preallocated SVM region. This is in contrast to when SVM is not
 * enabled, normal cl_mem allocation API is used also on the server side.
 */
int SharedCLContext::createBufferFromSVMRegion(BufferId_t BufferID, size_t Size,
                                               cl_mem_flags Flags,
                                               void *HostPtr,
                                               void **DeviceAddr) {

  // The backing drivers might not recognize the pocl PoC extension and
  // needn't as we implement the pinning with SVM allocations.
  Flags ^= CL_MEM_PINNED;
  bool SVMWrapper = false;
  SVMRegion *TargetSVMRegion = nullptr;

  // FIXME: The check doesn't account for the host-device SVM region offset.
  // The HostPtr that comes in is always a client/host address.
  if (HostPtr != nullptr &&
      ((size_t)HostPtr >= (size_t)SVMRegionsStartAddress &&
       (size_t)HostPtr < (size_t)SVMRegionsEndAddress)) {
    // This is a host-side pointer inside the SVM region, meaning a cl_mem
    // wrapped SVM pointer request. No need to allocate a new SVM host_ptr for
    // the internal buffer, just use the one already SVM allocated and wrap it
    // to a subbuffer.
    SVMWrapper = true;

    // Find the SVMRegion for it.
    for (auto &SVMRegion : SVMRegions)
      if (HostPtr >= SVMRegion.StartAddress &&
          HostPtr < (void *)((size_t)SVMRegion.StartAddress + SVMRegion.Size))
        TargetSVMRegion = &SVMRegion;

    if (TargetSVMRegion == nullptr) {
      POCL_MSG_ERR("Could not find the SVM region for given host ptr %p\n",
                   HostPtr);
      return CL_OUT_OF_RESOURCES;
    }

  } else {
#ifdef ENABLE_RDMA
    // TODO: What extra consideration we need for RDMA which uses SVM
    // also for the shadow buffers for the regular cl_mem?
    POCL_ABORT_UNIMPLEMENTED("SVM mode not yet implemented for PoCL-R RDMA.");
#endif

    // Allocate everything from the first SVM memory pool where it fits.
    chunk_info_t *Chunk = nullptr;
    for (auto &SVMRegion : SVMRegions) {
      Chunk = pocl_alloc_buffer_from_region(SVMRegion.Allocations, Size);
      if (Chunk != nullptr) {
        TargetSVMRegion = &SVMRegion;
        break;
      } else {
        POCL_MSG_PRINT_MEMORY("Didn't fit %zu to SVM region at %p...", Size,
                              SVMRegion.StartAddress);
      }
    }

    HostPtr = Chunk == nullptr ? nullptr : (void *)Chunk->start_address;

    assert(DeviceAddr != nullptr);
    *DeviceAddr = HostPtr;

    if (HostPtr == nullptr) {
      POCL_MSG_ERR("Error when creating a CG SVM allocation.");
      return CL_OUT_OF_RESOURCES;
    }
    POCL_MSG_PRINT_MEMORY(
        "Allocated a %zu byte chunk from an SVM region (start %p) at %p.\n",
        Size, TargetSVMRegion->StartAddress, HostPtr);
  }

  // Since the buffer data is initialized by clEnqueueWrite (because migration
  // handling is done in the client side), we cannot use any HOST_* flags here.
  Flags = Flags & (cl_bitfield)(CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY |
                                CL_MEM_READ_ONLY);

  // Allocate a subbuffer (server-side) from the parent SVM buffer for each
  // client-side allocation.

  cl_int Err;
  cl_buffer_region SubBufRegion;
  SubBufRegion.origin = (size_t)HostPtr - (size_t)TargetSVMRegion->StartAddress;
  SubBufRegion.size = Size;
  cl_mem SubBuf =
      clCreateSubBuffer(TargetSVMRegion->ShadowBuffer, Flags,
                        CL_BUFFER_CREATE_TYPE_REGION, &SubBufRegion, &Err);
  clBufferPtr Buf(new cl::Buffer(SubBuf));
  if (Err != CL_SUCCESS) {
    POCL_MSG_ERR(
        "P %u clCreateSubBuffer with origin %zx and size %zu failed %lu"
        " = %d\n",
        plat_id, SubBufRegion.origin, SubBufRegion.size, BufferID, Err);
    return Err;
  }

  {
    std::unique_lock<std::mutex> Lock(BufferMapMutex);
    BufferIDmap[BufferID] = std::move(Buf);
    if (!SVMWrapper) {
      // There can be multiple buffers pointing to the
      // same host ptr since cl_mem can wrap SVMs buffers.

      // For freeing client-side SVM allocations properly, we want to uniquely
      // identify its subbuffer to allow it to be found when the client calls
      // clSVMFree() with a raw SVM pointer.
      SVMShadowBufferIDMap[HostPtr] = BufferID;
    }
    SVMBackingStoreMap[BufferID] = HostPtr;
  }

  POCL_MSG_PRINT_MEMORY("P %u Created an SVM-Backed %sSubBuffer %lu"
                        " ptr %p\n",
                        plat_id, SVMWrapper ? "SVM Wrapper " : "", BufferID,
                        HostPtr);
  return 0;
}

int SharedCLContext::createBuffer(BufferId_t BufferID, size_t Size,
                                  cl_mem_flags Flags, void *HostPtr,
                                  void **DeviceAddr) {

  if (SVMRegions.size() > 0)
    return createBufferFromSVMRegion(BufferID, Size, Flags, HostPtr,
                                     DeviceAddr);

  // If there is a client-side host pointer in non-SVM mode, we should
  // not pass it to the server side buffer as it points to the client
  // memory.
  HostPtr = nullptr;

  // Since the buffer data is initialized by clEnqueueWrite (because migration
  // handling is done in the client side), we cannot use any HOST_* flags here.
  Flags = Flags & (cl_bitfield)(CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY |
                                CL_MEM_READ_ONLY);
  // when RDMA is used, VirtualClContext passes in a pointer from clSVMAlloc
  Flags = Flags |
          (cl_bitfield)(HostPtr ? CL_MEM_USE_HOST_PTR : CL_MEM_ALLOC_HOST_PTR);

  cl_int Err;
  clBufferPtr Buf(
      new cl::Buffer(ContextWithAllDevices, Flags, Size, HostPtr, &Err));
  if (Err != CL_SUCCESS) {
    POCL_MSG_ERR("P %u Create Buffer (size %zu, host ptr %p) failed %zu = %d\n",
                 plat_id, Size, HostPtr, BufferID, Err);
    return Err;
  }

  {
    std::unique_lock<std::mutex> Lock(BufferMapMutex);
    BufferIDmap[BufferID] = std::move(Buf);
    if (HostPtr)
      SVMBackingStoreMap[BufferID] = HostPtr;
  }

  POCL_MSG_PRINT_MEMORY("P %u Created Buffer %lu host_ptr %p\n", plat_id,
                        BufferID, HostPtr);
  return 0;
}

int SharedCLContext::freeBuffer(BufferId_t BufferId, bool IsSVMFree) {
  std::unique_lock<std::mutex> Lock(BufferMapMutex);

  if (SVMRegions.size() > 0) {

    void *DeviceSVMAddrToFree = nullptr;
    void *BufHostPtr = nullptr;
    if (IsSVMFree) {
      // Calling clSVMFree() directly from the client code: Free both the
      // bookkeeping subbuffer and the underlying SVM chunk.
      BufHostPtr = DeviceSVMAddrToFree = (void *)BufferId;
      POCL_MSG_PRINT_MEMORY("P %u SVMPool SVMFreeing raw SVM ptr at %p\n",
                            plat_id, BufHostPtr);

    } else {
      // A cl_mem free. If it's a SVMWrapper, we should just free the
      // subbuffer, but leave the backing SVM allocation alone.
      cl::Buffer *Buf = findBuffer(BufferId);
      assert(Buf != nullptr);

      BufHostPtr = Buf->getInfo<CL_MEM_HOST_PTR>();
      if (SVMShadowBufferIDMap.find(Buf->getInfo<CL_MEM_HOST_PTR>()) !=
              SVMShadowBufferIDMap.end() &&
          SVMShadowBufferIDMap[BufHostPtr] == BufferId) {
        DeviceSVMAddrToFree = Buf->getInfo<CL_MEM_HOST_PTR>();
      }
      POCL_MSG_PRINT_MEMORY(
          "P %u SVMPool freeing buffer id %lu (backing store at %p%s)\n",
          plat_id, BufferId, BufHostPtr,
          DeviceSVMAddrToFree != nullptr ? ", owner" : "");
    }

    if (!IsSVMFree && BufferIDmap.erase(BufferId) == 0) {
      POCL_MSG_ERR("Did not find the book keeping subbuffer to free at %p (buf "
                   "id %zu).\n",
                   BufHostPtr, BufferId);
    }

    if (DeviceSVMAddrToFree != nullptr) {
      // This is the SVM chunk "owner" (the initial SVM allocation).

      // TODO: apply host-device SVM offset adjustment
      memory_region_t *Region = nullptr;
      for (auto &SVMRegion : SVMRegions)
        if (SVMRegion.StartAddress <= DeviceSVMAddrToFree &&
            DeviceSVMAddrToFree <
                (void *)((size_t)SVMRegion.StartAddress + SVMRegion.Size))
          Region = SVMRegion.Allocations;

      memory_region_t *FoundRegion =
          pocl_free_buffer(Region, (memory_address_t)DeviceSVMAddrToFree);

      if (FoundRegion == nullptr || FoundRegion != Region) {
        POCL_MSG_ERR(
            "Did not find the SVM chunk to free at %p. A double free attempt?",
            DeviceSVMAddrToFree);
        return 0;
      }
      SVMShadowBufferIDMap.erase(BufHostPtr);
    }
    return 0;
  } else {
    POCL_MSG_PRINT_MEMORY("P %u Freeing a cl_mem buffer id %zu\n", plat_id,
                          BufferId);

    if (BufferIDmap.erase(BufferId) == 0) {
      POCL_MSG_ERR("P %u Unable to free cl_mem Buffer %" PRIu64 "\n", plat_id,
                   BufferId);
      return CL_INVALID_MEM_OBJECT;
    }
    // Free the possible RDMA backing store for the buffer.
    if (SVMBackingStoreMap.find(BufferId) != SVMBackingStoreMap.end()) {
      clSVMFree(ContextWithAllDevices.get(), SVMBackingStoreMap[BufferId]);
      SVMBackingStoreMap.erase(BufferId);
    }
  }
  return 0;
}

int SharedCLContext::migrateMemObject(uint64_t ev_id, uint32_t cq_id,
                                      uint32_t mem_obj_id, unsigned is_image,
                                      EventTiming_t &evt,
                                      uint32_t waitlist_size,
                                      uint64_t *waitlist) {
  cl::Buffer *b = nullptr;
  cl::Image *img = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Memory> vec{};
  POCL_MSG_PRINT_GENERAL("P %u Migrating %" PRIu32 " within Context\n", plat_id,
                         mem_obj_id);
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    if (!is_image) {
      uint32_t buffer_id = mem_obj_id;
      FIND_BUFFER;
      vec.push_back(*b);
    } else {
      uint32_t image_id = mem_obj_id;
      FIND_IMAGE;
      vec.push_back(*img);
    }
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);
  unsigned refcount = vec[0].getInfo<CL_MEM_REFERENCE_COUNT>();
  POCL_MSG_PRINT_GENERAL("memobj before migration: %u \n", refcount);

  //  EVENT_TIMING("migrateBuffer", cq->enqueueMigrateMemObjects(vec, 0,
  //  nullptr, &event));
  EVENT_TIMING_PRE;
  err = cq->enqueueMigrateMemObjects(vec, 0, &dependencies, &event);
  refcount = vec[0].getInfo<CL_MEM_REFERENCE_COUNT>();
  POCL_MSG_PRINT_GENERAL("memobj after migration: %u \n", refcount);
  EVENT_TIMING_POST("migrateBuffer");
}

int SharedCLContext::readBuffer(uint64_t ev_id, uint32_t cq_id,
                                uint64_t buffer_id, int is_svm,
                                uint32_t content_size_buffer_id, size_t size,
                                size_t offset, void *host_ptr,
                                uint64_t *out_size, EventTiming_t &evt,
                                uint32_t waitlist_size, uint64_t *waitlist) {

  cl::Buffer *b = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  if (!is_svm) {
    {
      FIND_BUFFER;
    }
    if (content_size_buffer_id != 0) {
      cl::Buffer *content_size = nullptr;
      FIND_BUFFER2(content_size);

      uint64_t content_bytes = 0;
      // TODO: blocks on all previous commands
      cq->enqueueReadBuffer(*content_size, CL_TRUE, 0, sizeof(content_bytes),
                            &content_bytes);
      POCL_MSG_PRINT_GENERAL("READ BUFFER SIZE %" PRIuS
                             " WITH CONTENT SIZE %" PRIu64 "\n",
                             size, content_bytes);
      if (offset > content_bytes)
        size = 0;
      else if (content_bytes < offset + size)
        size = content_bytes - offset;
    }

    if (out_size)
      *out_size = size;
    EVENT_TIMING("readBuffer",
                 cq->enqueueReadBuffer(*b, CL_FALSE, offset, size, host_ptr,
                                       &dependencies, &event));
  } else {
    void *svm_ptr = (void *)buffer_id;
    cl_int Err = clEnqueueSVMMap(cq->get(), CL_TRUE, CL_MAP_WRITE, svm_ptr,
                                 size, 0, NULL, NULL);

    if (Err != CL_SUCCESS) {
      POCL_MSG_ERR("Couldn't map SVM region at '%p' (size %zu).\n", svm_ptr,
                   size);
      return -1;
    }

    Err = clEnqueueSVMMemcpy(cq->get(), CL_TRUE, host_ptr, svm_ptr, size, 0,
                             NULL, NULL);
    if (Err != CL_SUCCESS) {
      POCL_MSG_ERR(
          "SVM memcpy failed when migrating data from '%p' (size %zu).\n",
          svm_ptr, size);
      return -1;
    }

    EVENT_TIMING("readBuffer (SVM)",
                 cq->enqueueUnmapSVM(svm_ptr, &dependencies, &event));
  }
}

int SharedCLContext::writeBuffer(uint64_t ev_id, uint32_t cq_id,
                                 uint64_t buffer_id, int is_svm, size_t size,
                                 size_t offset, void *host_ptr,
                                 EventTiming_t &evt, uint32_t waitlist_size,
                                 uint64_t *waitlist) {

  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  if (!is_svm) {

    cl::Buffer *b = nullptr;
    { FIND_BUFFER; }

    if (b->getInfo<CL_MEM_USES_SVM_POINTER>()) {
      void *buf_host_ptr = b->getInfo<CL_MEM_HOST_PTR>();
      // memcpy(buf_host_ptr, host_ptr, size);
      //  TODO: The host view is not updated with the below enqueueWriteBuffer.
      //  could that lead to race conditions if the server host view is synched
      //  wrongly to the GPU memory although we want the explict copies. For
      //  example after the writebuffer there would be an implicit unmap which
      //  would then overwrite the device view with whatever is in the server
      //  host memory.
    }

#if 1
    EVENT_TIMING("writeBuffer",
                 cq->enqueueWriteBuffer(*b, CL_TRUE, offset, size, host_ptr,
                                        &dependencies, &event));
#endif

    // cq->finish();
  } else {

    // TODO: we should not get writebuffer requests with is_svm on anymore.
    // this branch is obsolete.
    assert("We should not request SVM copies direcrly anymore." && false);

    // TODO: SVM updates should be written directly to the (host mapped) SVM
    // region when reading the data from network. Now there's an extra copy!

    // Map the region to host first so we can update it. buffer_id is the SVM
    // device pointer instead of a cl_mem.

    // TODO: we could use async ops? But will the request
    // extra_data buffer be alive until the commands get finished?

    void *device_svm_ptr = (void *)buffer_id;
    cl_int Err = clEnqueueSVMMap(cq->get(), CL_TRUE, CL_MAP_WRITE,
                                 device_svm_ptr, size, 0, NULL, NULL);

    if (Err != CL_SUCCESS) {
      POCL_MSG_ERR("Couldn't map SVM region at '%p' (size %zu).\n",
                   device_svm_ptr, size);
      return -1;
    }

    Err = clEnqueueSVMMemcpy(cq->get(), CL_TRUE, device_svm_ptr, host_ptr, size,
                             0, NULL, NULL);
    if (Err != CL_SUCCESS) {
      POCL_MSG_ERR(
          "SVM memcpy failed when migrating data at '%p' (size %zu).\n",
          device_svm_ptr, size);
      return -1;
    }

    EVENT_TIMING("writeBuffer (SVM)",
                 cq->enqueueUnmapSVM(device_svm_ptr, &dependencies, &event));
  }
  return 0;
}

int SharedCLContext::copyBuffer(uint64_t ev_id, uint32_t cq_id,
                                uint32_t src_buffer_id, uint32_t dst_buffer_id,
                                uint32_t content_size_buffer_id, size_t size,
                                size_t src_offset, size_t dst_offset,
                                EventTiming_t &evt, uint32_t waitlist_size,
                                uint64_t *waitlist) {
  cl::Buffer *src = nullptr;
  cl::Buffer *dst = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_BUFFER2(src);
    FIND_BUFFER2(dst);
  }
  if (content_size_buffer_id != 0) {
    cl::Buffer *content_size = nullptr;
    FIND_BUFFER2(content_size);

    uint64_t content_bytes = 0;
    // TODO: blocks on all previous commands
    cq->enqueueReadBuffer(*content_size, CL_TRUE, 0, sizeof(content_bytes),
                          &content_bytes);
    POCL_MSG_PRINT_GENERAL("READ BUFFER SIZE %" PRIuS
                           " WITH CONTENT SIZE %" PRIu64 "\n",
                           size, content_bytes);
    if (src_offset > content_bytes)
      size = 0;
    else if (content_bytes < src_offset + size)
      size = content_bytes - src_offset;
  }

  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);
  if (size != 0) {
    EVENT_TIMING("copyBuffer",
                 cq->enqueueCopyBuffer(*src, *dst, src_offset, dst_offset, size,
                                       &dependencies, &event));
  } else {
    // Zero sized copy is not allowed, just use a marker as a stand-in for event
    // sync purposes
    EVENT_TIMING("copyBuffer",
                 cq->enqueueMarkerWithWaitList(&dependencies, &event));
  }
}

int SharedCLContext::readBufferRect(
    uint64_t ev_id, uint32_t cq_id, uint32_t buffer_id,
    sizet_vec3 &buffer_origin, sizet_vec3 &region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, void *host_ptr, size_t host_bytes,
    EventTiming_t &evt, uint32_t waitlist_size, uint64_t *waitlist) {
  cl::Buffer *b = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_BUFFER;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("readBufferRect",
               cq->enqueueReadBufferRect(*b, CL_FALSE, buffer_origin,
                                         zero_origin, region, buffer_row_pitch,
                                         buffer_slice_pitch, 0, 0, host_ptr,
                                         &dependencies, &event));
}

int SharedCLContext::writeBufferRect(
    uint64_t ev_id, uint32_t cq_id, uint32_t buffer_id,
    sizet_vec3 &buffer_origin, sizet_vec3 &region, size_t buffer_row_pitch,
    size_t buffer_slice_pitch, void *host_ptr, size_t host_bytes,
    EventTiming_t &evt, uint32_t waitlist_size, uint64_t *waitlist) {
  cl::Buffer *b = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_BUFFER;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("writeBufferRect",
               cq->enqueueWriteBufferRect(*b, CL_FALSE, buffer_origin,
                                          zero_origin, region, buffer_row_pitch,
                                          buffer_slice_pitch, 0, 0, host_ptr,
                                          &dependencies, &event));
}

int SharedCLContext::copyBufferRect(
    uint64_t ev_id, uint32_t cq_id, uint32_t dst_buffer_id,
    uint32_t src_buffer_id, sizet_vec3 &dst_origin, sizet_vec3 &src_origin,
    sizet_vec3 &region, size_t dst_row_pitch, size_t dst_slice_pitch,
    size_t src_row_pitch, size_t src_slice_pitch, EventTiming_t &evt,
    uint32_t waitlist_size, uint64_t *waitlist) {
  cl::Buffer *src = nullptr;
  cl::Buffer *dst = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_BUFFER2(src);
    FIND_BUFFER2(dst);
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("copyBufferRect",
               cq->enqueueCopyBufferRect(*src, *dst, src_origin, dst_origin,
                                         region, src_row_pitch, src_slice_pitch,
                                         dst_row_pitch, dst_slice_pitch,
                                         nullptr, &event));
}

#define fillB(type)                                                            \
  {                                                                            \
    type *patt = reinterpret_cast<type *>(pattern);                            \
    err =                                                                      \
        cq->enqueueFillBuffer(*b, *patt, offset, size, &dependencies, &event); \
    break;                                                                     \
  }

int SharedCLContext::fillBuffer(uint64_t ev_id, uint32_t cq_id,
                                uint32_t buffer_id, size_t offset, size_t size,
                                void *pattern, size_t pattern_size,
                                EventTiming_t &evt, uint32_t waitlist_size,
                                uint64_t *waitlist) {
  cl::Buffer *b = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_BUFFER;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING_PRE;

  switch (pattern_size) {
  case 1:
    fillB(cl_uchar);
  case 2:
    fillB(cl_ushort);
  case 3:
    fillB(cl_uchar3);
  case 4:
    fillB(cl_uint);
  case 6:
    fillB(cl_ushort3);
  case 8:
    fillB(cl_ulong);
  case 12:
    fillB(cl_uint3);
  case 16:
    fillB(cl_ulong2);
  case 24:
    fillB(cl_ulong3);
  case 32:
    fillB(cl_ulong4);
  case 64:
    fillB(cl_ulong8);
  default:
    err = CL_INVALID_ARG_VALUE;
  }

  EVENT_TIMING_POST("fillBuffer");
}

int SharedCLContext::setKernelArgs(cl::Kernel *k, clKernelStruct *kernel,
                                   size_t arg_count, uint64_t *args,
                                   unsigned char *is_svm_ptr, size_t pod_size,
                                   char *pod_buf) {
  cl_int err;

  assert(arg_count == kernel->numArgs);

  if (arg_count == 0)
    return CL_SUCCESS;

  const char *pod_tmp = pod_buf;
  {
    for (cl_uint i = 0; i < arg_count; ++i) {

      switch (kernel->metaData->arg_meta[i].type) {

      case PoclRemoteArgType::Local: {
        POCL_MSG_PRINT_GENERAL("Setting ARG %u type Local \n", i);
        cl::size_type size = args[i];
        err = k->setArg(i, size, nullptr);
        assert(err == CL_SUCCESS);
        break;
      }

      case PoclRemoteArgType::Image: {
        uint32_t img_id = static_cast<uint32_t>(args[i]);
        POCL_MSG_PRINT_GENERAL("Setting ARG %u type IMAGE  image id: %" PRIu32
                               " ARGS[i]: %" PRIu64 " \n",
                               i, img_id, args[i]);
        cl::Image *img = findImage(img_id);
        assert(img);
        err = k->setArg<>(i, (*img));
        assert(err == CL_SUCCESS);
        break;
      }
      case PoclRemoteArgType::Sampler: {
        uint32_t samp_id = static_cast<uint32_t>(args[i]);
        POCL_MSG_PRINT_GENERAL(
            "Setting ARG %u type SAMPLER  sampler id: %" PRIu32
            " ARGS[i]: %" PRIu64 "\n",
            i, samp_id, args[i]);
        cl::Sampler *samp = findSampler(samp_id);
        assert(samp);
        err = k->setArg<>(i, (*samp));
        assert(err == CL_SUCCESS);
        break;
      }
      case PoclRemoteArgType::Pointer: {
        if (is_svm_ptr[i]) {
          void *svm_ptr = (void *)(args[i]);
          POCL_MSG_PRINT_GENERAL("Setting ARG %u type POINTER (SVM), %p\n", i,
                                 svm_ptr);
          err = ::clSetKernelArgSVMPointer(k->get(), i, svm_ptr);
          if (err != CL_SUCCESS) {
            POCL_MSG_ERR(
                "SVM pointer arg %d could not be set to '%p' error code: %d.\n",
                i, svm_ptr, err);
          }
          // Assert in a server based on input data is a bit... smelly.
          assert(err == CL_SUCCESS);
        } else if (kernel->metaData->arg_meta[i].address_qualifier ==
                   CL_KERNEL_ARG_ADDRESS_LOCAL) {
          POCL_MSG_PRINT_GENERAL(
              "Setting ARG %u type POINTER (LOCAL), size: %" PRIu64 "\n", i,
              args[i]);
          err = k->setArg(i, static_cast<size_t>(args[i]), nullptr);
          assert(err == CL_SUCCESS);
        } else {
          uint32_t buffer_id = static_cast<uint32_t>(args[i]);
          POCL_MSG_PRINT_GENERAL(
              "Setting ARG %u type POINTER, buffer id: %" PRIu32
              " ARGS[i]: %" PRIu64 "\n",
              i, buffer_id, args[i]);
          if (buffer_id == 0) {
            POCL_MSG_WARN("NULL PTR ARG DETECTED: KERNEL %s ARG %u / %s \n",
                          kernel->metaData->meta.name, i,
                          kernel->metaData->arg_meta[i].name);
            err = k->setArg(i, cl::Buffer());
            assert(err == CL_SUCCESS);
          } else {
            cl::Buffer *b = findBuffer(buffer_id);
            assert(b);
            err = k->setArg<>(i, (*b));
            assert(err == CL_SUCCESS);
          }
        }
        break;
      }
      case PoclRemoteArgType::POD: {
        cl::size_type size = args[i];
        if (size == 4) {
          int32_t jjj = *(int32_t *)pod_tmp;
          POCL_MSG_PRINT_GENERAL(
              "Setting ARG %u type POD to int32_t: %" PRId32 " \n", i, jjj);
        } else
          POCL_MSG_PRINT_GENERAL(
              "Setting ARG %u type POD to size: %" PRIuS " \n", i, size);
        if (size > 0) {
          err = k->setArg(i, size, (const void *)pod_tmp);
          assert(err == CL_SUCCESS);
          pod_tmp += size;
        }
        break;
      }
      }
    }
  }

  POCL_MSG_PRINT_GENERAL("DONE SETTING ARGS\n");

  return CL_SUCCESS;
}

int SharedCLContext::runKernel(
    uint64_t ev_id, uint32_t cq_id, uint32_t device_id, uint16_t has_new_args,
    size_t arg_count, uint64_t *args, unsigned char *is_svm_ptr,
    size_t pod_size, char *pod_buf, EventTiming_t &evt, uint32_t kernel_id,
    uint32_t waitlist_size, uint64_t *waitlist, unsigned dim,
    const sizet_vec3 &offset, const sizet_vec3 &global,
    const sizet_vec3 *local) {
  cl::Kernel *k = nullptr;
  clKernelStruct *kernel = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_KERNEL;
  }

  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING_PRE;
  cl::NDRange o(offset[0], offset[1], offset[2]);
  // required because work dimensions are determined from global_size.
  cl::NDRange g1(global[0]);
  cl::NDRange g2(global[0], global[1]);
  cl::NDRange g3(global[0], global[1], global[2]);

  std::unique_lock<std::mutex> kernelLock(kernel->Lock);
  if (has_new_args) {
    int r = setKernelArgs(k, kernel, arg_count, args, is_svm_ptr, pod_size,
                          pod_buf);
    assert(r == CL_SUCCESS);
  }

  {
    std::unique_lock<std::mutex> Lock(BufferMapMutex);
    std::vector<void *> SVMPtrs;
#if 0
    // Do we need to pass the SVM regions here? It will kill the perf.
    // as the regions are large.
    // TODO: We should pass the indirect SVM
    // pointers from the client-side, but they are not passed either.
    if (SVMRegions.size() > 0)
      SVMPtrs.push_back(SVMPool);
#endif
    for (auto &S : SVMBackingStoreMap)
      SVMPtrs.push_back(S.second);
    k->setSVMPointers(SVMPtrs);
  }
  {
    err = cq->enqueueNDRangeKernel(
        *k, o, (dim == 2 ? g2 : (dim < 2 ? g1 : g3)),
        ((local == nullptr)
             ? cl::NullRange
             : cl::NDRange((*local)[0], (*local)[1], (*local)[2])),
        &dependencies, &event);
  }
  {
    std::unique_lock<std::mutex> lock(EventmapMutex);
    auto map_result = Eventmap.insert({ev_id, {event, cl::UserEvent()}});
    if (!map_result.second) {
      map_result.first->second.native = event;
    }
  }

  if (err == CL_SUCCESS)
    POCL_MSG_PRINT_EVENTS("NDRangeKernel: ID %" PRIu32 ", CQ: %" PRIu32
                          " event: %" PRIu64 " \n",
                          kernel_id, cq_id, ev_id);
  else {
    int e;
    std::string name = k->getInfo<CL_KERNEL_FUNCTION_NAME>(&e);
    assert(e == CL_SUCCESS);

    POCL_MSG_ERR("enqueue NDRangeKernel failed: %" PRIu32 " / %s, CQ: %" PRIu32
                 " ERR: %d\n",
                 kernel_id, name.c_str(), cq_id, err);
    return err;
  }

  return err;
}

/***************************************************************************/
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

int SharedCLContext::fillImage(uint64_t ev_id, uint32_t cq_id,
                               uint32_t image_id, sizet_vec3 &origin,
                               sizet_vec3 &region, void *fill_color,
                               EventTiming_t &evt, uint32_t waitlist_size,
                               uint64_t *waitlist) {
  cl::Image *img = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_IMAGE;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  cl_uint4 fillColor = *(cl_uint4 *)fill_color;
  EVENT_TIMING("fillImage",
               cq->enqueueFillImage(*img, fillColor, origin, region,
                                    &dependencies, &event));
}

// readImage2Buffer(request.id, m.dst_buf_id, origin, region)
int SharedCLContext::copyImage2Buffer(uint64_t ev_id, uint32_t cq_id,
                                      uint32_t image_id, uint32_t buffer_id,
                                      sizet_vec3 &origin, sizet_vec3 &region,
                                      size_t offset, EventTiming_t &evt,
                                      uint32_t waitlist_size,
                                      uint64_t *waitlist) {

  cl::Buffer *b = nullptr;
  cl::Image *img = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_IMAGE;
    FIND_BUFFER;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("copyImage2Buffer",
               cq->enqueueCopyImageToBuffer(*img, *b, origin, region, offset,
                                            &dependencies, &event));
}

// writeBuffer2Image(request.id, m.src_buf_id, origin, region)
int SharedCLContext::copyBuffer2Image(uint64_t ev_id, uint32_t cq_id,
                                      uint32_t image_id, uint32_t buffer_id,
                                      sizet_vec3 &origin, sizet_vec3 &region,
                                      size_t offset, EventTiming_t &evt,
                                      uint32_t waitlist_size,
                                      uint64_t *waitlist) {
  cl::Buffer *b = nullptr;
  cl::Image *img = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_IMAGE;
    FIND_BUFFER;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("copyBuffer2Image",
               cq->enqueueCopyBufferToImage(*b, *img, offset, origin, region,
                                            &dependencies, &event));
}

int SharedCLContext::copyImage2Image(uint64_t ev_id, uint32_t cq_id,
                                     uint32_t dst_image_id,
                                     uint32_t src_image_id,
                                     sizet_vec3 &dst_origin,
                                     sizet_vec3 &src_origin, sizet_vec3 &region,
                                     EventTiming_t &evt, uint32_t waitlist_size,
                                     uint64_t *waitlist) {

  cl::Image *src = nullptr;
  cl::Image *dst = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_IMAGE2(src);
    FIND_IMAGE2(dst);
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("copyImage2Image",
               cq->enqueueCopyImage(*src, *dst, src_origin, dst_origin, region,
                                    &dependencies, &event));
}

int SharedCLContext::readImageRect(uint64_t ev_id, uint32_t cq_id,
                                   uint32_t image_id, sizet_vec3 &origin,
                                   sizet_vec3 &region, void *host_ptr,
                                   size_t host_bytes, EventTiming_t &evt,
                                   uint32_t waitlist_size, uint64_t *waitlist) {
  cl::Image *img = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_IMAGE;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("readImageRect",
               cq->enqueueReadImage(*img, CL_FALSE, origin, region, 0, 0,
                                    host_ptr, &dependencies,
                                    &event)); // TODO row pitch / slice pitch
}

int SharedCLContext::writeImageRect(uint64_t ev_id, uint32_t cq_id,
                                    uint32_t image_id, sizet_vec3 &origin,
                                    sizet_vec3 &region, void *host_ptr,
                                    size_t host_bytes, EventTiming_t &evt,
                                    uint32_t waitlist_size,
                                    uint64_t *waitlist) {
  cl::Image *img = nullptr;
  cl::CommandQueue *cq = nullptr;
  std::vector<cl::Event> dependencies;
  {
    FIND_QUEUE;
    FIND_IMAGE;
  }
  dependencies = remapWaitlist(waitlist_size, waitlist, ev_id);

  EVENT_TIMING("writeImageRect",
               cq->enqueueWriteImage(*img, CL_FALSE, origin, region, 0, 0,
                                     host_ptr, &dependencies,
                                     &event)); // TODO row pitch / slice pitch
}

/***************************************************************************/
/***************************************************************************/

SharedContextBase *createSharedCLContext(cl::Platform *platform, size_t pid,
                                         VirtualContextBase *v,
                                         ReplyQueueThread *slow,
                                         ReplyQueueThread *fast) {
  SharedCLContext *clctx = new SharedCLContext(platform, pid, v, slow, fast);
  return clctx;
}
