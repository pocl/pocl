/* level0-driver.hh - driver for LevelZero Compute API devices.

   Copyright (c) 2022-2023 Michal Babej / Intel Finland Oy

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

#ifndef LEVEL0DRIVER_HH
#define LEVEL0DRIVER_HH

#include <ze_api.h>
#include <pocl_cl.h>

#include "level0-compilation.hh"

namespace pocl {

#define LEVEL0_CHECK_RET(RETVAL, CODE)                                         \
  do {                                                                         \
    ze_result_t res = CODE;                                                    \
    if (res != ZE_RESULT_SUCCESS) {                                            \
      POCL_MSG_PRINT2(ERROR, __FUNCTION__, __LINE__,                           \
                      "Error %i from Level0 Runtime call:\n", (int)res);       \
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

class Level0WorkQueueInterface {
public:
  virtual void pushWork(_cl_command_node *Command) = 0;
  virtual _cl_command_node *getWorkOrWait(bool &ShouldExit) = 0;
};

class Level0Queue {
  ze_command_queue_handle_t QueueH;
  ze_command_list_handle_t CmdListH;
  uint64_t *EventStart = nullptr;
  uint64_t *EventFinish = nullptr;

  std::thread Thread;
  Level0WorkQueueInterface *WorkHandler;

  uint64_t HostTimingStart;
  uint64_t DeviceTimingStart;
  double HostDeviceRate;
  uint32_t_3 DeviceMaxWGSizes;
  bool DeviceHasGlobalOffsets;

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
                const size_t *__restrict__ const DstOrigin,
                const size_t *__restrict__ const SrcOrigin,
                const size_t *__restrict__ const Region,
                size_t const DstRowPitch,
                size_t const DstSlicePitch,
                size_t const SrcRowPitch,
                size_t const SrcSlicePitch);
  void readRect(void *__restrict__ HostVoidPtr,
                pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
                const size_t *__restrict__ const BufferOrigin,
                const size_t *__restrict__ const HostOrigin,
                const size_t *__restrict__ const Region,
                size_t const BufferRowPitch,
                size_t const BufferSlicePitch,
                size_t const HostRowPitch,
                size_t const HostSlicePitch);
  void writeRect(const void *__restrict__ HostVoidPtr,
                 pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                 const size_t *__restrict__ const BufferOrigin,
                 const size_t *__restrict__ const HostOrigin,
                 const size_t *__restrict__ const Region,
                 size_t const BufferRowPitch,
                 size_t const BufferSlicePitch,
                 size_t const HostRowPitch,
                 size_t const HostSlicePitch);
  void memFill(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
               size_t Size, size_t Offset,
               const void *__restrict__ Pattern,
               size_t PatternSize);
  void mapMem(pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
              mem_mapping_t *Map);
  void unmapMem(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                mem_mapping_t *map);

  /* copies image to image, on the same device (or same global memory). */
  void copyImageRect(cl_mem SrcImage, cl_mem DstImage,
                     pocl_mem_identifier *SrcMemId,
                     pocl_mem_identifier *DstMemId,
                     const size_t *SrcOrigin,
                     const size_t *DstOrigin,
                     const size_t *Region);

  /* copies a region from host OR device buffer to device image.
   * clEnqueueCopyBufferToImage: src_mem_id = buffer,
   *     src_host_ptr = NULL, src_row_pitch = src_slice_pitch = 0
   * clEnqueueWriteImage: src_mem_id = NULL,
   *     src_host_ptr = host pointer, src_offset = 0
   */
  void writeImageRect(cl_mem DstImage,
                      pocl_mem_identifier *DstMemId,
                      const void *__restrict__ src_HostPtr,
                      pocl_mem_identifier *SrcMemId,
                      const size_t *Origin, const size_t *Region,
                      size_t SrcRowPitch, size_t SrcSlicePitch,
                      size_t SrcOffset);

  /* copies a region from device image to host or device buffer
   * clEnqueueCopyImageToBuffer: dst_mem_id = buffer,
   *     dst_host_ptr = NULL, dst_row_pitch = dst_slice_pitch = 0
   * clEnqueueReadImage: dst_mem_id = NULL,
   *     dst_host_ptr = host pointer, dst_offset = 0
   */
  void readImageRect(cl_mem SrcImage,
                     pocl_mem_identifier *SrcMemId,
                     void *__restrict__ DstHostPtr,
                     pocl_mem_identifier *DstMemId,
                     const size_t *Origin, const size_t *Region,
                     size_t DstRowPitch, size_t DstSlicePitch,
                     size_t DstOffset);

  /* maps the entire image from device to host */
  void mapImage(pocl_mem_identifier *MemId, cl_mem SrcImage,
                mem_mapping_t *Map);

  /* unmaps the entire image from host to device */
  void unmapImage(pocl_mem_identifier *MemId, cl_mem DstImage,
                  mem_mapping_t *Map);

  /* fill image with pattern */
  void fillImage(cl_mem Image, pocl_mem_identifier *MemId,
                 const size_t *Origin, const size_t *Region,
                 cl_uint4 OrigPixel, pixel_t FillPixel,
                 size_t PixelSize);

  void svmMap(void* Ptr);
  void svmUnmap(void* Ptr);
  void svmCopy(void* DstPtr, const void* SrcPtr, size_t Size);
  void svmFill(void *DstPtr, size_t Size, void* Pattern, size_t PatternSize);

  bool setupKernelArgs(ze_module_handle_t ModuleH, ze_kernel_handle_t KernelH,
                       cl_device_id Dev, unsigned DeviceI,
                       _cl_command_run *RunCmd);
  void runWithOffsets(struct pocl_context *PoclCtx, ze_kernel_handle_t KernelH);
  void run(_cl_command_node *Cmd);

  void execCommand(_cl_command_node *Cmd);

  void syncUseMemHostPtr(pocl_mem_identifier *MemId, cl_mem Mem,
                         size_t Offset, size_t Size);
  void syncUseMemHostPtr(pocl_mem_identifier *MemId, cl_mem Mem,
                         size_t Origin[3], size_t Region[3],
                         size_t RowPitch, size_t SlicePitch);
  void calculateEventTimes(cl_event Event, uint64_t Start, uint64_t Finish);

public:
  Level0Queue(Level0WorkQueueInterface *WH, ze_command_queue_handle_t Q,
              ze_command_list_handle_t L, uint64_t *TimestampBuffer,
              double HostDevRate, uint64_t HostTimeStart,
              uint64_t DeviceTimeStart, uint32_t_3 DeviceMaxWGs,
              bool DeviceHasGOffsets);
  ~Level0Queue();

  void runThread();
};

class Level0QueueGroup : public Level0WorkQueueInterface {
  std::vector<std::unique_ptr<Level0Queue>> Queues;

  std::condition_variable Cond __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));
  std::mutex Mutex  __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));

  std::queue<_cl_command_node *> WorkQueue __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));
  bool ThreadExitRequested;
  uint64_t *TimeStampBuffer;


public:
  Level0QueueGroup(){};
  ~Level0QueueGroup();
  bool init(unsigned Ordinal, unsigned Count, ze_context_handle_t ContextH,
            ze_device_handle_t DeviceH, uint64_t *Buffer,
            uint64_t HostTimingStart, uint64_t DeviceTimingStart,
            double HostDeviceRate, uint32_t_3 DeviceMaxWGs,
            bool DeviceHasGOffsets);

  virtual void pushWork(_cl_command_node *Command);
  virtual _cl_command_node *getWorkOrWait(bool &ShouldExit);
};


class Level0Driver;

class Level0Device {
  Level0QueueGroup CopyQueues;
  Level0QueueGroup ComputeQueues;
  // TODO check reliability
  ze_device_uuid_t UUID;
  // TODO: it seems libze just returs zeroes for KernelUUID
  ze_native_kernel_uuid_t KernelUUID;
  std::string KernelCacheHash;
  cl_device_id ClDev;
  ze_device_handle_t DeviceHandle;
  ze_context_handle_t ContextHandle;
  Level0Driver *Driver;
  bool Available = false;
  bool Integrated = false;
  bool OndemandPaging = false;
  bool Supports64bitBuffers = false;
  uint32_t MaxCommandQueuePriority = 0;
  uint32_t TSBits = 0;
  uint32_t KernelTSBits = 0;
  uint32_t MaxWGCount[3];
  uint32_t MaxMemoryFillPatternSize = 0;
  uint32_t GlobalMemOrd = UINT32_MAX;
  uint64_t *CopyTimestamps;
  uint64_t *ComputeTimestamps;

public:
  Level0Device(Level0Driver *Drv, ze_device_handle_t DeviceH,
               ze_context_handle_t ContextH,
               cl_device_id dev, const char *Parameters);
  ~Level0Device();

  void pushCommand(_cl_command_node *Command);

  void *allocSharedMem(uint64_t Size);
  void *allocDeviceMem(uint64_t Size);
  void freeMem(void *Ptr);

  ze_image_handle_t allocImage(cl_channel_type ChType,
                               cl_channel_order ChOrder,
                               cl_mem_object_type ImgType,
                               cl_mem_flags ImgFlags, size_t Width,
                               size_t Height, size_t Depth);
  void freeImage(ze_image_handle_t ImageH);

  ze_sampler_handle_t allocSampler(cl_addressing_mode AddrMode,
                                   cl_filter_mode FilterMode,
                                   cl_bool NormalizedCoords);
  void freeSampler(ze_sampler_handle_t SamplerH);

  int createProgram(cl_program Program, cl_uint DeviceI);
  int freeProgram(cl_program Program, cl_uint DeviceI);
};

typedef std::unique_ptr<Level0Device> Level0DeviceUPtr;

class Level0Driver {
  ze_driver_handle_t DriverH = nullptr;
  std::vector<ze_device_handle_t> DeviceHandles;
  std::set<std::string> ExtensionSet;
  std::vector<Level0DeviceUPtr> Devices;
  ze_context_handle_t ContextH = nullptr;
  // TODO: doesn't seem reliably the same between runs
  ze_driver_uuid_t UUID;
  uint32_t Version = 0;
  unsigned NumDevices = 0;
  Level0CompilationJobScheduler JobSched;

public:
  Level0Driver();
  ~Level0Driver();
  unsigned getNumDevices() { return Devices.size(); }
  const ze_driver_uuid_t &getUUID() { return UUID; }
  uint32_t getVersion() { return Version; }
  Level0Device *createDevice(unsigned Index, cl_device_id Dev, const char *Params);
  void releaseDevice(Level0Device *Dev);
  bool hasExtension(const char *Name) {
    return ExtensionSet.find(Name) != ExtensionSet.end();
  }
  bool empty() { return NumDevices == 0; }
  Level0CompilationJobScheduler &getJobSched() { return JobSched; }
};

} // namespace pocl

#endif // LEVEL0DRIVER_HH
