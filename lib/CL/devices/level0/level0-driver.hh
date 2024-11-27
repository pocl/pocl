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
              ze_command_list_handle_t L, Level0Device *D,
              size_t MaxPatternSize);
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
  uint32_t MaxFillPatternSize;

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
  void readRectHelper(char *HostPtr, const char *DevicePtr,
                      const size_t *BufferOrigin, const size_t *HostOrigin,
                      const size_t *Region, size_t const BufferRowPitch,
                      size_t const BufferSlicePitch, size_t const HostRowPitch,
                      size_t const HostSlicePitch);
  void readRect(void *__restrict__ HostVoidPtr, pocl_mem_identifier *SrcMemId,
                cl_mem SrcBuf, const size_t *__restrict__ BufferOrigin,
                const size_t *__restrict__ HostOrigin,
                const size_t *__restrict__ Region, size_t BufferRowPitch,
                size_t BufferSlicePitch, size_t HostRowPitch,
                size_t HostSlicePitch);
  void writeRectHelper(const char *HostPtr, char *DevicePtr,
                       const size_t *BufferOrigin, const size_t *HostOrigin,
                       const size_t *Region, size_t const BufferRowPitch,
                       size_t const BufferSlicePitch, size_t const HostRowPitch,
                       size_t const HostSlicePitch);
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

  void runBuiltinKernel(_cl_command_run *RunCmd, cl_device_id Dev,
                        cl_event Event, cl_program Program, cl_kernel Kernel,
                        unsigned DeviceI);
  void runNDRangeKernel(_cl_command_run *RunCmd, cl_device_id Dev,
                        cl_event Event, cl_program Program, cl_kernel Kernel,
                        unsigned DeviceI, pocl_buffer_migration_info *MigInfos);

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

  bool init(unsigned Ordinal, unsigned Count, Level0Device *Device,
            size_t MaxPatternSize);
  void uninit();

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

class Level0Allocator {
public:
  virtual void *allocBuffer(uintptr_t Key, Level0Device *D,
                            ze_device_mem_alloc_flags_t DevFlags,
                            ze_host_mem_alloc_flags_t HostFlags, size_t Size,
                            bool &IsHostAccessible) = 0;
  virtual bool freeBuffer(uintptr_t Key, Level0Device *D, void *Ptr) = 0;
  virtual bool clear(Level0Device *D) = 0;
};

using Level0AllocatorSPtr = std::shared_ptr<Level0Allocator>;

class Level0Device {

public:
  Level0Device(Level0Driver *Drv, ze_device_handle_t DeviceH,
               cl_device_id Dev, const char *Parameters);
  ~Level0Device();

  Level0Device(Level0Device const &) = delete;
  Level0Device& operator=(Level0Device const &) = delete;
  Level0Device(Level0Device const &&) = delete;
  Level0Device& operator=(Level0Device &&) = delete;

  void pushCommand(_cl_command_node *Command);
  void pushCommandBatch(BatchType Batch);

  void assignAllocator(Level0AllocatorSPtr NewAlloc) { Alloc = NewAlloc; }
  void *allocBuffer(uintptr_t Key, ze_device_mem_alloc_flags_t DevFlags,
                    ze_host_mem_alloc_flags_t HostFlags, size_t Size,
                    bool &IsHostAccessible) {
    return Alloc->allocBuffer(Key, this, DevFlags, HostFlags, Size,
                              IsHostAccessible);
  }
  bool freeBuffer(uintptr_t Key, void *Ptr) {
    return Alloc->freeBuffer(Key, this, Ptr);
  }

  void *allocUSMSharedMem(uint64_t Size, bool EnableCompression = false,
                          ze_device_mem_alloc_flags_t DevFlags =
                              ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,
                          ze_host_mem_alloc_flags_t HostFlags =
                              ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED |
                              ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT |
                              ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
  void *allocUSMDeviceMem(uint64_t Size,
                          ze_device_mem_alloc_flags_t DevFlags =
                              ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED);
  void *allocUSMHostMem(uint64_t Size,
                        ze_device_mem_alloc_flags_t HostFlags =
                            ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,
                        void *pNext = nullptr);
  void freeUSMMem(void *Ptr);
  bool freeUSMMemBlocking(void *Ptr);

  ze_image_handle_t allocImage(cl_channel_type ChType,
                               cl_channel_order ChOrder,
                               cl_mem_object_type ImgType,
                               cl_mem_flags ImgFlags, size_t Width,
                               size_t Height, size_t Depth, size_t ArraySize);
  static void freeImage(ze_image_handle_t ImageH);

  ze_sampler_handle_t allocSampler(cl_addressing_mode AddrMode,
                                   cl_filter_mode FilterMode,
                                   cl_bool NormalizedCoords);
  static void freeSampler(ze_sampler_handle_t SamplerH);

  int createSpirvProgram(cl_program Program, cl_uint DeviceI);
  int createBuiltinProgram(cl_program Program, cl_uint DeviceI);
  int freeProgram(cl_program Program, cl_uint DeviceI);

  int createKernel(cl_program Program, cl_kernel Kernel,
                   unsigned ProgramDeviceI);
  int freeKernel(cl_program Program, cl_kernel Kernel, unsigned ProgramDeviceI);

  bool getBestKernel(Level0Program *Program, Level0Kernel *Kernel,
                     bool LargeOffset, unsigned LocalWGSize,
                     ze_module_handle_t &Mod, ze_kernel_handle_t &Ker);

#ifdef ENABLE_NPU
  bool getBestBuiltinKernel(Level0BuiltinProgram *Program,
                            Level0BuiltinKernel *Kernel,
                            ze_graph_handle_t &Graph);
#endif

  bool getMemfillKernel(unsigned PatternSize, Level0Kernel **L0Kernel,
                        ze_module_handle_t &ModH, ze_kernel_handle_t &KerH);

  bool getImagefillKernel(cl_channel_type ChType,
                          cl_channel_order ChOrder,
                          cl_mem_object_type ImgType,
                          Level0Kernel **L0Kernel,
                          ze_module_handle_t &ModH,
                          ze_kernel_handle_t &KerH);

  const std::vector<size_t> &getSupportedSubgroupSizes() {
    return SupportedSubgroupSizes;
  }
  cl_bitfield getMemCaps(cl_device_info Type);
  cl_unified_shared_memory_type_intel getMemType(const void *USMPtr);
  void *getMemBasePtr(const void *USMPtr);
  size_t getMemSize(const void *USMPtr);
  cl_device_id getMemAssoc(const void *USMPtr);
  cl_mem_alloc_flags_intel getMemFlags(const void *USMPtr);

  ze_event_handle_t getNewEvent();
  ze_device_handle_t getDeviceHandle() { return DeviceHandle; }
  ze_context_handle_t getContextHandle() { return ContextHandle; }
  Level0CompilationJobScheduler &getJobSched();
  Level0Driver *getDriver() const { return Driver; }
  cl_device_id getClDev();
  void getTimingInfo(uint32_t &TS, uint32_t &KernelTS, double &TimerFreq,
                     double &NsPerCycle);
  void getMaxWGs(uint32_t_3 *MaxWGs);
  uint32_t getMaxWGSize() { return ClDev->max_work_group_size; }
  // max WorkGroup size for a particular Kernel
  uint32_t getMaxWGSizeForKernel(Level0Kernel *Kernel);
  // max SubGroup size for a particular Kernel
  uint32_t getMaxSGSizeForKernel(Level0Kernel *Kernel) {
    // TODO we should get the real value from the L0 API somehow
    return 8;
  }
  bool isHostUnifiedMemory() { return ClDev->host_unified_memory; }
  bool supportsHostUSM() { return ClDev->host_usm_capabs != 0; }
  bool supportsDeviceUSM() { return ClDev->device_usm_capabs != 0; }
  bool supportsSingleSharedUSM() {
    return ClDev->single_shared_usm_capabs != 0;
  }
  bool supportsCrossSharedUSM() {
    return ClDev->cross_shared_usm_capabs != 0;
  }
  bool supportsSystemSharedUSM() {
    return ClDev->system_shared_usm_capabs != 0;
  }
  bool supportsOndemandPaging() { return OndemandPaging; }
  bool supportsGlobalOffsets() { return HasGOffsets; }
  bool supportsCompression() { return HasCompression; }
  bool supportsExportByDmaBuf() { return HasDMABufExport; }
  bool supportsImportByDmaBuf() { return HasDMABufImport; }
  const ze_device_properties_t &getProperties() { return DeviceProperties; }
  cl_device_feature_capabilities_intel getFeatureCaps() {
    cl_device_feature_capabilities_intel FeatureCaps = 0;
    if (SupportsDPAS)
      FeatureCaps |= CL_DEVICE_FEATURE_FLAG_DPAS_INTEL;
    if (SupportsDP4A)
      FeatureCaps |= CL_DEVICE_FEATURE_FLAG_DP4A_INTEL;
    return FeatureCaps;
  }
  uint32_t getIPVersion() { return DeviceIPVersion; }

  bool supportsCmdQBatching() {
    return UniversalQueues.available() && ClDev->type == CL_DEVICE_TYPE_GPU;
  }
  // for GPU, prefer L0 queues for all commands, as most commands can be
  // implemented using L0 API calls, and the few that can't (e.g. for
  // imagefill) we have implemented via kernels
  bool prefersZeQueues() { return ClDev->type == CL_DEVICE_TYPE_GPU; }
  // NPU allocates memory with L0 Host type, and many commands can't be
  // implemented with L0 API calls because the linux-npu-driver does not
  // support "user pointers" (memory other than allocated by driver).
  // This makes it difficult to support command lists since we would
  // have to analyze every command to see if it uses any "user pointers"
  // and memcpy the memory before/after the command. Additionally also
  // every command in a batch would have to be checked, and if necessary
  // the batch needs to be split at points where memcpy is required.
  bool prefersHostQueues() { return ClDev->type == CL_DEVICE_TYPE_CUSTOM; }

private:
  Level0AllocatorSPtr Alloc;
  std::deque<Level0EventPool> EventPools;
  std::mutex EventPoolLock;
  Level0QueueGroup CopyQueues;
  Level0QueueGroup ComputeQueues;
  Level0QueueGroup UniversalQueues;

  std::map<std::string, Level0Kernel *> MemfillKernels;
  std::map<std::string, Level0Kernel *> ImagefillKernels;

  Level0Driver *Driver;
  cl_device_id ClDev;
  ze_device_handle_t DeviceHandle;
  ze_context_handle_t ContextHandle;
  std::string Extensions;
  std::string OpenCL30Features;
  std::string BuiltinKernels;
  unsigned NumBuiltinKernels = 0;
  // need to store for queries
  ze_device_properties_t DeviceProperties;
  uint32_t DeviceIPVersion;

  Level0Program *MemfillProgram;
  Level0Program *ImagefillProgram;

  // TODO: it seems libze just returs zeroes for KernelUUID
  ze_native_kernel_uuid_t KernelUUID;
  std::string KernelCacheHash;

  cl_bool Available = CL_FALSE;
  bool OndemandPaging = false;
  bool Supports64bitBuffers = false;
  bool SupportsDP4A = false;
  bool SupportsDPAS = false;
  bool NeedsRelaxedLimits = false;
  bool HasGOffsets = false;
  bool HasCompression = false;
  bool HasDMABufExport = false;
  bool HasDMABufImport = false;
  uint32_t MaxCommandQueuePriority = 0;
  uint32_t TSBits = 0;
  uint32_t KernelTSBits = 0;
  double TimerNsPerCycle = 0.0;
  double TimerFrequency = 0.0;
  uint32_t MaxWGCount[3];
  uint32_t MaxMemoryFillPatternSize = 0;
  uint32_t GlobalMemOrd = UINT32_MAX;
  std::vector<size_t> SupportedSubgroupSizes;

  /// initializes kernels used internally by the driver
  /// to implement functionality missing in the Level Zero API,
  /// e.g. FillImage, FillBuffer with large patterns etc
  bool initHelperKernels();
  void destroyHelperKernels();

  bool setupDeviceProperties(bool HasIPVersionExt);
  bool setupComputeProperties();
  bool setupModuleProperties(bool &SupportsInt64Atomics, bool HasFloatAtomics, std::string &Features);
  bool setupQueueGroupProperties();
  bool setupMemoryProperties(bool &HasUSMCapability);
  void setupGlobalMemSize(bool HasRelaxedAllocLimits);
  bool setupCacheProperties();
  bool setupImageProperties();
  bool setupPCIAddress();
};

typedef std::unique_ptr<Level0Device> Level0DeviceUPtr;

class Level0Driver {

public:
  Level0Driver(ze_driver_handle_t DrvHandle);
  ~Level0Driver();

  Level0Driver(Level0Driver const &) = delete;
  Level0Driver& operator=(Level0Driver const &) = delete;
  Level0Driver(Level0Driver const &&) = delete;
  Level0Driver& operator=(Level0Driver &&) = delete;

  ze_context_handle_t getContextHandle() { return ContextH; }
#ifdef ENABLE_NPU
  graph_dditable_ext_t *getGraphExt() { return GraphDDITableExt; }
#endif
  unsigned getNumDevices() { return Devices.size(); }
  Level0Device *getExportDevice();
  bool getImportDevices(std::vector<Level0Device *> &ImportDevices,
                        Level0Device *ExcludeDev);
  const ze_driver_uuid_t *getUUID() { return &UUID; }
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
  std::vector<Level0DeviceUPtr> Devices;
  std::set<std::string> ExtensionSet;
  std::map<ze_device_handle_t, cl_device_id> HandleToIDMap;
  ze_context_handle_t ContextH = nullptr;

#ifdef ENABLE_NPU
  /// @brief Pointer to the Level Zero API graph extension DDI table.
  graph_dditable_ext_t *GraphDDITableExt = nullptr;
  /// @brief Pointer to the Level Zero API graph extension profiling DDI table.
  ze_graph_profiling_dditable_ext_t *GraphProfDDITableExt = nullptr;
#endif

  // TODO: doesn't seem reliably the same between runs
  ze_driver_uuid_t UUID;
  uint32_t Version = 0;
  unsigned NumDevices = 0;
  Level0CompilationJobScheduler JobSched;
};

using Level0DriverUPtr = std::unique_ptr<Level0Driver>;

class Level0DefaultAllocator : public Level0Allocator {
public:
  Level0DefaultAllocator(Level0Driver *Dr, Level0Device *Dev)
      : Driver(Dr), Device(Dev) {};

  // the default allocator ignores the Device argument, since there is only one
  virtual void *allocBuffer(uintptr_t Key, Level0Device *,
                            ze_device_mem_alloc_flags_t DevFlags,
                            ze_host_mem_alloc_flags_t HostFlags, size_t Size,
                            bool &IsHostAccessible) override;
  virtual bool freeBuffer(uintptr_t Key, Level0Device *, void *Ptr) override;
  virtual bool clear(Level0Device *D) override { return true; }

private:
  Level0Driver *Driver;
  Level0Device *Device;
};

using Level0DefaultAllocatorUPtr = std::unique_ptr<Level0DefaultAllocator>;

/// manages multiple device allocations for a single buffer
/// automatically allocates export memory first and releases it last
class DMABufAllocation {
public:
  DMABufAllocation() = default;
  DMABufAllocation(const DMABufAllocation &) = default;
  DMABufAllocation(DMABufAllocation &&) = default;
  DMABufAllocation &operator=(DMABufAllocation &&) = default;
  DMABufAllocation &operator=(const DMABufAllocation &) = default;
  ~DMABufAllocation();

  void *allocExport(Level0Device *D, ze_device_mem_alloc_flags_t DevFlags,
                    ze_host_mem_alloc_flags_t HostFlags, size_t Size);
  void *allocImport(Level0Device *D, ze_device_mem_alloc_flags_t DevFlags,
                    ze_host_mem_alloc_flags_t HostFlags, size_t Size);
  bool free(Level0Device *D);
  bool isValid() { return FD >= 0; }

private:
  using DevicePtrMap = std::map<Level0Device *, void *>;
  DevicePtrMap BufferImportMap;

  Level0Device *ExportDev = nullptr;
  void *ExportPtr = nullptr;
  int FD = -1;
};

/// multi-L0-device (and multi-L0-context) allocator,
/// using DMABUF export/import L0 API
/// to create cross-device zero-copy allocations
class Level0DMABufAllocator : public Level0Allocator {
public:
  Level0DMABufAllocator(Level0Device *ExDev,
                        const std::vector<Level0Device *> &ImDev)
      : ImportDevices(ImDev), ExportDevice(ExDev) {};

  virtual void *allocBuffer(uintptr_t Key, Level0Device *D,
                            ze_device_mem_alloc_flags_t DevFlags,
                            ze_host_mem_alloc_flags_t HostFlags, size_t Size,
                            bool &IsHostAccessible) override;
  virtual bool freeBuffer(uintptr_t Key, Level0Device *D, void *Ptr) override;
  virtual bool clear(Level0Device *D) override;

private:
  /// single L0 device that supports export via DMABUF;
  /// this is used to create the first allocation and File Descriptor
  Level0Device *ExportDevice;
  /// vector of devices that will allocate the buffer by importing the FD
  std::vector<Level0Device *> ImportDevices;
  /// Key can be anything but currently we're using cl_mem
  std::map<uintptr_t, DMABufAllocation> Allocations;
};

using Level0DMABufAllocatorSPtr = std::shared_ptr<Level0DMABufAllocator>;

} // namespace pocl

#endif // POCL_LIB_CL_DEVICES_LEVEL0_LEVEL0_DRIVER_HH
