/// level0-driver.hh - driver for LevelZero Compute API devices.
///
/// Copyright (c) 2022-2023 Michal Babej / Intel Finland Oy
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.


#ifndef POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_DRIVER_HH
#define POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_DRIVER_HH

#include <pocl_cl.h>
#include <ze_api.h>

#include "level0-compilation.hh"

namespace pocl {

#define LEVEL0_CHECK_RET(RETVAL, CODE)                                         \
  do {                                                                         \
    ze_result_t res = CODE;                                                    \
    if (res != ZE_RESULT_SUCCESS) {                                            \
      POCL_MSG_PRINT2(ERROR, __FUNCTION__, __LINE__,                           \
                      "Error %0x from Level0 Runtime call:\n", (int)res);      \
      return RETVAL;                                                           \
    }                                                                          \
  } while (0)

#ifndef HAVE_UINT32_T_3
#define HAVE_UINT32_T_3
typedef struct
{
  uint32_t s[3];
} uint32_t_3;
#endif

using BatchType = std::deque<cl_event>;
/// limit the Batch size to this number of commands
constexpr unsigned BatchSizeLimit = 128;
/// the number of events allocated for each Event Pool
constexpr unsigned EventPoolSize = 2048;

class Level0WorkQueueInterface {

public:
  virtual void pushWork(_cl_command_node *Command) = 0;
  virtual void pushCommandBatch(BatchType Batch) = 0;
  virtual bool getWorkOrWait(_cl_command_node **Node, BatchType &Batch) = 0;
  virtual ~Level0WorkQueueInterface() {};
};

class Level0Device;

class Level0Queue {

public:
  Level0Queue(Level0WorkQueueInterface *WH, ze_command_queue_handle_t Q,
              ze_command_list_handle_t L,
              Level0Device *D);
  ~Level0Queue();

  Level0Queue(Level0Queue const &) = delete;
  Level0Queue& operator=(Level0Queue const &) = delete;
  Level0Queue(Level0Queue const &&) = delete;
  Level0Queue& operator=(Level0Queue &&) = delete;

  void runThread();

private:
  std::queue<ze_event_handle_t> AvailableDeviceEvents;
  std::queue<ze_event_handle_t> DeviceEventsToReset;
  std::map<void *, size_t> MemPtrsToMakeResident;
  std::map<std::pair<char*, char*>, size_t> UseMemHostPtrsToSync;

  ze_command_queue_handle_t QueueH;
  ze_command_list_handle_t CmdListH;

  ze_event_handle_t CurrentEventH;
  ze_event_handle_t PreviousEventH;

  Level0Device *Device;
  std::thread Thread;
  Level0WorkQueueInterface *WorkHandler;

  double DeviceFrequency;
  double DeviceNsPerCycle;
  // maximum valid (kernel) timestamp value
  uint64_t DeviceMaxValidTimestamp;
  uint64_t DeviceMaxValidKernelTimestamp;
  // Nanoseconds after which the device (kernel) timer wraps around
  uint64_t DeviceTimerWrapTimeNs;
  uint64_t DeviceKernelTimerWrapTimeNs;
  uint32_t_3 DeviceMaxWGSizes;

  void read(void *__restrict__ HostPtr,
            pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
            size_t Offset, size_t Size);
  void write(const void *__restrict__ HostPtr,
             pocl_mem_identifier *DstMemId, cl_mem DstBuf,
             size_t Offset, size_t Size);
  void copy(pocl_mem_identifier *DstMemDd, cl_mem DstBuf,
            pocl_mem_identifier *SrcMemId, cl_mem SrcBuf, size_t DstOffset,
            size_t SrcOffset, size_t Size);
  void copyRect(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
                const size_t *__restrict__ DstOrigin,
                const size_t *__restrict__ SrcOrigin,
                const size_t *__restrict__ Region, size_t DstRowPitch,
                size_t DstSlicePitch, size_t SrcRowPitch, size_t SrcSlicePitch);
  void readRect(void *__restrict__ HostVoidPtr, pocl_mem_identifier *SrcMemId,
                cl_mem SrcBuf, const size_t *__restrict__ BufferOrigin,
                const size_t *__restrict__ HostOrigin,
                const size_t *__restrict__ Region, size_t BufferRowPitch,
                size_t BufferSlicePitch, size_t HostRowPitch,
                size_t HostSlicePitch);
  void writeRect(const void *__restrict__ HostVoidPtr,
                 pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                 const size_t *__restrict__ BufferOrigin,
                 const size_t *__restrict__ HostOrigin,
                 const size_t *__restrict__ Region, size_t BufferRowPitch,
                 size_t BufferSlicePitch, size_t HostRowPitch,
                 size_t HostSlicePitch);
  void memFill(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
               size_t Size, size_t Offset,
               const void *__restrict__ Pattern,
               size_t PatternSize);
  void memfillImpl(Level0Device *Device, ze_command_list_handle_t CmdListH,
                   const void *MemPtr, size_t Size, size_t Offset,
                   const void *__restrict__ Pattern, size_t PatternSize);
  void mapMem(pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
              mem_mapping_t *Map);
  void unmapMem(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                mem_mapping_t *Map);

  void copyImageRect(cl_mem SrcImage, cl_mem DstImage,
                     pocl_mem_identifier *SrcMemId,
                     pocl_mem_identifier *DstMemId,
                     const size_t *SrcOrigin,
                     const size_t *DstOrigin,
                     const size_t *Region);

  void writeImageRect(cl_mem DstImage,
                      pocl_mem_identifier *DstMemId,
                      const void *__restrict__ src_HostPtr,
                      pocl_mem_identifier *SrcMemId,
                      const size_t *Origin, const size_t *Region,
                      size_t SrcRowPitch, size_t SrcSlicePitch,
                      size_t SrcOffset);

  void readImageRect(cl_mem SrcImage,
                     pocl_mem_identifier *SrcMemId,
                     void *__restrict__ DstHostPtr,
                     pocl_mem_identifier *DstMemId,
                     const size_t *Origin, const size_t *Region,
                     size_t DstRowPitch, size_t DstSlicePitch,
                     size_t DstOffset);

  void mapImage(pocl_mem_identifier *MemId, cl_mem SrcImage,
                mem_mapping_t *Map);

  void unmapImage(pocl_mem_identifier *MemId, cl_mem DstImage,
                  mem_mapping_t *Map);

  void fillImage(cl_mem Image, pocl_mem_identifier *MemId,
                 const size_t *Origin, const size_t *Region,
                 cl_uint4 OrigPixel, pixel_t FillPixel,
                 size_t PixelSize);

  static void svmMap(void *Ptr);
  static void svmUnmap(void *Ptr);
  void svmCopy(void* DstPtr, const void* SrcPtr, size_t Size);
  void svmFill(void *DstPtr, size_t Size, void* Pattern, size_t PatternSize);
  void svmMigrate(unsigned num_svm_pointers, void **svm_pointers,
                  size_t *sizes);
  void svmAdvise(const void *ptr, size_t size, cl_mem_advice_intel advice);

  bool setupKernelArgs(ze_module_handle_t ModuleH, ze_kernel_handle_t KernelH,
                       cl_device_id Dev, unsigned DeviceI,
                       _cl_command_run *RunCmd);
  void runWithOffsets(struct pocl_context *PoclCtx, ze_kernel_handle_t KernelH);
  void run(_cl_command_node *Cmd);

  void appendEventToList(_cl_command_node *Cmd, const char **Msg);
  void execCommand(_cl_command_node *Cmd);
  void execCommandBatch(BatchType &Batch);
  void reset();
  void closeCmdList();
  void makeMemResident();
  void syncMemHostPtrs();
  void allocNextFreeEvent();

  void syncUseMemHostPtr(pocl_mem_identifier *MemId, cl_mem Mem,
                         size_t Offset, size_t Size);
  void syncUseMemHostPtr(pocl_mem_identifier *MemId, cl_mem Mem,
                         const size_t Origin[3], const size_t Region[3],
                         size_t RowPitch, size_t SlicePitch);
};

class Level0QueueGroup : public Level0WorkQueueInterface {

public:
  Level0QueueGroup() {};
  ~Level0QueueGroup() override;

  Level0QueueGroup(Level0QueueGroup const &) = delete;
  Level0QueueGroup& operator=(Level0QueueGroup const &) = delete;
  Level0QueueGroup(Level0QueueGroup const &&) = delete;
  Level0QueueGroup& operator=(Level0QueueGroup &&) = delete;

  bool init(unsigned Ordinal, unsigned Count, Level0Device *Device);

  void pushWork(_cl_command_node *Command) override;
  void pushCommandBatch(BatchType Batch) override;

  bool getWorkOrWait(_cl_command_node **Node, BatchType &Batch) override;
  bool available() const { return Available; }

private:
  std::condition_variable Cond;
  std::mutex Mutex;

  std::queue<_cl_command_node *> WorkQueue;
  std::queue<BatchType> BatchWorkQueue;

  std::vector<std::unique_ptr<Level0Queue>> Queues;

  bool ThreadExitRequested = false;
  bool Available = false;
};

class Level0Driver;
class Level0Device;

class Level0EventPool {
public:
  Level0EventPool(Level0Device *D, unsigned EvtPoolSize);
  ~Level0EventPool();
  bool isEmpty() const { return LastIdx >= AvailableEvents.size(); }
  ze_event_handle_t getEvent();
private:
  std::vector<ze_event_handle_t> AvailableEvents;
  ze_event_pool_handle_t EvtPoolH;
  Level0Device *Dev;
  unsigned LastIdx;
};

class Level0Device {

public:
  Level0Device(Level0Driver *Drv, ze_device_handle_t DeviceH,
               cl_device_id dev, const char *Parameters);
  ~Level0Device();

  Level0Device(Level0Device const &) = delete;
  Level0Device& operator=(Level0Device const &) = delete;
  Level0Device(Level0Device const &&) = delete;
  Level0Device& operator=(Level0Device &&) = delete;

  void pushCommand(_cl_command_node *Command);
  void pushCommandBatch(BatchType Batch);

  void *allocSharedMem(uint64_t Size, bool EnableCompression = false,
                       ze_device_mem_alloc_flags_t DevFlags =
                           ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,
                       ze_host_mem_alloc_flags_t HostFlags =
                           ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED);
  void *allocDeviceMem(uint64_t Size, ze_device_mem_alloc_flags_t DevFlags =
                                          ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED);
  void *allocHostMem(uint64_t Size, ze_device_mem_alloc_flags_t HostFlags =
                                        ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED);
  void freeMem(void *Ptr);
  bool freeMemBlocking(void *Ptr);

  ze_image_handle_t allocImage(cl_channel_type ChType,
                               cl_channel_order ChOrder,
                               cl_mem_object_type ImgType,
                               cl_mem_flags ImgFlags, size_t Width,
                               size_t Height, size_t Depth);
  static void freeImage(ze_image_handle_t ImageH);

  ze_sampler_handle_t allocSampler(cl_addressing_mode AddrMode,
                                   cl_filter_mode FilterMode,
                                   cl_bool NormalizedCoords);
  static void freeSampler(ze_sampler_handle_t SamplerH);

  int createProgram(cl_program Program, cl_uint DeviceI);
  int freeProgram(cl_program Program, cl_uint DeviceI);
  const std::vector<size_t> &getSupportedSubgroupSizes() {
    return SupportedSubgroupSizes;
  }
  bool getBestKernel(Level0Program *Program, Level0Kernel *Kernel,
                     bool LargeOffset, unsigned LocalWGSize,
                     ze_module_handle_t &Mod, ze_kernel_handle_t &Ker);

  bool getMemfillKernel(unsigned PatternSize, Level0Kernel **L0Kernel,
                        ze_module_handle_t &ModH, ze_kernel_handle_t &KerH);

  bool getImagefillKernel(cl_channel_type ChType,
                          cl_channel_order ChOrder,
                          cl_mem_object_type ImgType,
                          Level0Kernel **L0Kernel,
                          ze_module_handle_t &ModH,
                          ze_kernel_handle_t &KerH);

  cl_bitfield getMemCaps(cl_device_info Type);
  cl_unified_shared_memory_type_intel getMemType(const void *USMPtr);
  void *getMemBasePtr(const void *USMPtr);
  size_t getMemSize(const void *USMPtr);
  cl_device_id getMemAssoc(const void *USMPtr);
  cl_mem_alloc_flags_intel getMemFlags(const void *USMPtr);

  ze_event_handle_t getNewEvent();
  ze_device_handle_t getDeviceHandle() { return DeviceHandle; }
  ze_context_handle_t getContextHandle() { return ContextHandle; }
  void getTimingInfo(uint32_t &TS, uint32_t &KernelTS, double &TimerFreq,
                     double &NsPerCycle);
  void getMaxWGs(uint32_t_3 *MaxWGs);
  uint32_t getMaxWGSize() { return ClDev->max_work_group_size; }
  bool supportsHostUSM() { return HostMemCaps != 0; }
  bool supportsDeviceUSM() { return DeviceMemCaps != 0; }
  bool supportsSingleSharedUSM() { return SingleSharedCaps != 0; }
  bool supportsCrossSharedUSM() { return CrossSharedCaps != 0; }
  bool supportsSystemSharedUSM() { return SystemSharedCaps != 0; }
  bool supportsOndemandPaging() { return OndemandPaging; }
  bool supportsGlobalOffsets() { return HasGOffsets; }
  bool supportsCompression() { return HasCompression; }
  bool supportsUniversalQueues() { return UniversalQueues.available(); }

private:
  std::deque<Level0EventPool> EventPools;
  std::mutex EventPoolLock;
  Level0QueueGroup CopyQueues;
  Level0QueueGroup ComputeQueues;
  Level0QueueGroup UniversalQueues;

  std::map<std::string, Level0Kernel *> MemfillKernels;
  std::map<std::string, Level0Kernel *> ImagefillKernels;

  Level0Program *MemfillProgram;
  Level0Program *ImagefillProgram;

  // TODO check reliability
  ze_device_uuid_t UUID;
  // TODO: it seems libze just returs zeroes for KernelUUID
  ze_native_kernel_uuid_t KernelUUID;
  std::string KernelCacheHash;
  cl_device_id ClDev;
  ze_device_handle_t DeviceHandle;
  ze_context_handle_t ContextHandle;
  Level0Driver *Driver;
  cl_bool Available = CL_FALSE;
  bool Integrated = false;
  bool OndemandPaging = false;
  bool Supports64bitBuffers = false;
  bool NeedsRelaxedLimits = false;
  bool HasGOffsets = false;
  bool HasCompression = false;
  uint32_t MaxCommandQueuePriority = 0;
  uint32_t TSBits = 0;
  uint32_t KernelTSBits = 0;
  double TimerNsPerCycle = 0.0;
  double TimerFrequency = 0.0;
  uint32_t MaxWGCount[3];
  uint32_t MaxMemoryFillPatternSize = 0;
  uint32_t GlobalMemOrd = UINT32_MAX;
  std::vector<size_t> SupportedSubgroupSizes;
  cl_device_unified_shared_memory_capabilities_intel HostMemCaps = 0;
  cl_device_unified_shared_memory_capabilities_intel DeviceMemCaps = 0;
  cl_device_unified_shared_memory_capabilities_intel SingleSharedCaps = 0;
  cl_device_unified_shared_memory_capabilities_intel CrossSharedCaps = 0;
  cl_device_unified_shared_memory_capabilities_intel SystemSharedCaps = 0;

  /// initializes kernels used internally by the driver
  /// to implement functionality missing in the Level Zero API,
  /// e.g. FillImage, FillBuffer with large patterns etc
  bool initHelperKernels();
  void destroyHelperKernels();
};

typedef std::unique_ptr<Level0Device> Level0DeviceUPtr;

class Level0Driver {

public:
  Level0Driver();
  ~Level0Driver();

  Level0Driver(Level0Driver const &) = delete;
  Level0Driver& operator=(Level0Driver const &) = delete;
  Level0Driver(Level0Driver const &&) = delete;
  Level0Driver& operator=(Level0Driver &&) = delete;

  ze_context_handle_t getContextHandle() { return ContextH; }
  unsigned getNumDevices() { return Devices.size(); }
  const ze_driver_uuid_t &getUUID() { return UUID; }
  uint32_t getVersion() const { return Version; }
  Level0Device *createDevice(unsigned Index, cl_device_id Dev, const char *Params);
  void releaseDevice(Level0Device *Dev);
  bool hasExtension(const char *Name) {
    return ExtensionSet.find(Name) != ExtensionSet.end();
  }
  bool empty() const { return NumDevices == 0; }
  Level0CompilationJobScheduler &getJobSched() { return JobSched; }
  cl_device_id getClDevForHandle(ze_device_handle_t H) {
    return HandleToIDMap[H];
  }

private:
  ze_driver_handle_t DriverH = nullptr;
  std::vector<ze_device_handle_t> DeviceHandles;
  std::set<std::string> ExtensionSet;
  std::vector<Level0DeviceUPtr> Devices;
  std::map<ze_device_handle_t, cl_device_id> HandleToIDMap;
  ze_context_handle_t ContextH = nullptr;
  // TODO: doesn't seem reliably the same between runs
  ze_driver_uuid_t UUID;
  uint32_t Version = 0;
  unsigned NumDevices = 0;
  Level0CompilationJobScheduler JobSched;
};

} // namespace pocl

#endif // POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_DRIVER_HH
