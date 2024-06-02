/* almaif.cc - generic/example driver for hardware accelerators with memory
   mapped control.

   Copyright (c) 2019-2020 Pekka Jääskeläinen / Tampere University
                 2022 Topi Leppänen / Tampere University

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

#include "almaif.h"
#include "AlmaIFRegion.hh"
#include "MMAPDevice.hh"
#include "config.h"

#ifdef HAVE_XRT
#include "XilinxXrtDevice.hh"
#define HAVE_DBDEVICE
#endif

#ifdef HAVE_DBDEVICE
#include "AlmaifDB/AlmaIFBitstreamDatabaseManager.hh"
#include "AlmaifDB/DBDevice.hh"
#endif

#include "EmulationDevice.hh"

#ifdef TCE_AVAILABLE
#include "openasip/TTASimDevice.hh"
#endif

#include "AlmaifCompile.hh"
#include "AlmaifShared.hh"
#include "bufalloc.h"
#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "openasip/AlmaifCompileOpenasip.hh"
#include "pocl_cl.h"
#include "pocl_timing.h"
#include "pocl_util.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <climits>
#include <iostream>
#include <set>
#include <string>
#include <vector>

extern int pocl_offline_compile;

class SimpleSimulatorFrontend;

//#define ALMAIF_DUMP_MEMORY
pocl_lock_t globalMemIDLock = POCL_LOCK_INITIALIZER;
bool isGlobalMemIDSet = false;
int GlobalMemID;

pocl_lock_t runningLock = POCL_LOCK_INITIALIZER;
pocl_lock_t runningDeviceLock = POCL_LOCK_INITIALIZER;
int runningDeviceCount = 0;
pocl_thread_t runningThread;
void *runningThreadFunc(void *);
_cl_command_node *runningList;
bool runningJoinRequested = false;

struct almaif_event_data_t {
  pthread_cond_t event_cond;
  chunk_info_t *chunk;
};

void pocl_almaif_init_device_ops(struct pocl_device_ops *ops) {

  ops->device_name = "almaif";
  ops->init = pocl_almaif_init;
  ops->uninit = pocl_almaif_uninit;
  ops->probe = pocl_almaif_probe;
  ops->build_hash = pocl_almaif_build_hash;
  ops->setup_metadata = pocl_setup_builtin_metadata;

  /* TODO: Bufalloc-based allocation from the onchip memories. */
  ops->alloc_mem_obj = pocl_almaif_alloc_mem_obj;
  ops->free = pocl_almaif_free;
  ops->write = pocl_almaif_write;
  ops->read = pocl_almaif_read;
  ops->copy = pocl_almaif_copy;

  ops->map_mem = pocl_almaif_map_mem;
  ops->unmap_mem = pocl_almaif_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->submit = pocl_almaif_submit;
  ops->join = pocl_almaif_join;
  ops->notify = pocl_almaif_notify;
  ops->broadcast = pocl_broadcast;
  ops->run = pocl_almaif_run;
  //  ops->update_event = pocl_almaif_update_event;
  ops->free_event_data = pocl_almaif_free_event_data;
  ops->update_event = pocl_almaif_update_event;

  ops->wait_event = pocl_almaif_wait_event;
  ops->notify_event_finished = pocl_almaif_notify_event_finished;
  ops->notify_cmdq_finished = pocl_almaif_notify_cmdq_finished;
  ops->init_queue = pocl_almaif_init_queue;
  ops->free_queue = pocl_almaif_free_queue;

  ops->build_builtin = pocl_driver_build_opencl_builtins;
  ops->free_program = pocl_driver_free_program;

  ops->copy_rect = pocl_almaif_copy_rect;
  ops->read_rect = pocl_almaif_read_rect;
  ops->write_rect = pocl_almaif_write_rect;

  ops->memfill = pocl_almaif_memfill;
}

void pocl_almaif_write(void *data, const void *__restrict__ src_host_ptr,
                      pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                      size_t offset, size_t size) {
  AlmaifData *d = (AlmaifData *)data;

  d->Dev->writeDataToDevice(dst_mem_id, (const char *__restrict)src_host_ptr,
                            size, offset);
}

void pocl_almaif_read(void *data, void *__restrict__ dst_host_ptr,
                     pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                     size_t offset, size_t size) {
  AlmaifData *d = (AlmaifData *)data;

  d->Dev->readDataFromDevice((char *__restrict__)dst_host_ptr, src_mem_id, size,
                             offset);
}

void pocl_almaif_copy(void *data, pocl_mem_identifier *dst_mem_id,
                     cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                     cl_mem src_buf, size_t dst_offset, size_t src_offset,
                     size_t size) {

  chunk_info_t *src_chunk = (chunk_info_t *)src_mem_id->mem_ptr;
  chunk_info_t *dst_chunk = (chunk_info_t *)dst_mem_id->mem_ptr;
  size_t src = src_chunk->start_address + src_offset;
  size_t dst = dst_chunk->start_address + dst_offset;
  if (src == dst) {
    return;
  }
  AlmaifData *d = (AlmaifData *)data;

  if (d->Dev->DataMemory->isInRange(dst)) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes from %zx to 0x%zx\n", size,
                         src, dst);
    if (d->Dev->DataMemory->isInRange(src)) {
      d->Dev->DataMemory->CopyInMem(src, dst, size);
    } else {
      char *__restrict__ src_ptr = (char *)src_mem_id->mem_ptr;
      d->Dev->DataMemory->CopyToMMAP(dst, src_ptr + src_offset, size);
    }
  } else if (d->Dev->ExternalMemory && d->Dev->ExternalMemory->isInRange(dst)) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes to external 0x%zx\n", size,
                         dst);
    d->Dev->ExternalMemory->CopyInMem(src, dst, size);

  } else {
    POCL_ABORT("Attempt to copy data outside the device memories\n");
  }
}

void pocl_almaif_memfill(void *data, pocl_mem_identifier *dst_mem_id,
                      cl_mem dst_buf, size_t size, size_t offset,
                      const void *__restrict__ pattern, size_t pattern_size) {
  void *tmp_memfill_buf = pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, size);
  assert(tmp_memfill_buf);
  pocl_fill_aligned_buf_with_pattern(tmp_memfill_buf, 0, size, pattern,
                                     pattern_size);
  pocl_almaif_write(data, tmp_memfill_buf, dst_mem_id, dst_buf, offset, size);

  POCL_MEM_FREE(tmp_memfill_buf);
}


cl_int pocl_almaif_alloc_mem_obj(cl_device_id device, cl_mem mem_obj,
                                void *host_ptr) {

  AlmaifData *data = (AlmaifData *)device->data;

  cl_int alloc_success = CL_SUCCESS;
  if (mem_obj->type == CL_MEM_OBJECT_PIPE) {
    pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];
    alloc_success = data->Dev->allocatePipe(p, mem_obj->size);
  } else {
    /* almaif driver doesn't preallocate */
    if ((mem_obj->flags & CL_MEM_ALLOC_HOST_PTR) &&
        (mem_obj->mem_host_ptr == NULL))
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;

    pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];
    alloc_success = data->Dev->allocateBuffer(p, mem_obj->size);
  }

  return alloc_success;
}


void pocl_almaif_free(cl_device_id device, cl_mem mem) {

  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  AlmaifData *data = (AlmaifData *)device->data;

  if (mem->type == CL_MEM_OBJECT_PIPE) {
    data->Dev->freePipe(p);
  } else {
    data->Dev->freeBuffer(p);
  }

  p->mem_ptr = NULL;
  p->version = 0;

}

cl_int pocl_almaif_map_mem(void *data, pocl_mem_identifier *src_mem_id,
                          cl_mem src_buf, mem_mapping_t *map) {

  /* Synch the device global region to the host memory. */
  pocl_almaif_read(data, map->host_ptr, src_mem_id, src_buf, map->offset,
                  map->size);

  return CL_SUCCESS;
}

cl_int pocl_almaif_unmap_mem(void *data, pocl_mem_identifier *dst_mem_id,
                            cl_mem dst_buf, mem_mapping_t *map) {

  if (map->map_flags != CL_MAP_READ) {
  /* Synch the host memory to the device global region. */
  pocl_almaif_write(data, map->host_ptr, dst_mem_id, dst_buf, map->offset,
                  map->size);
  }

  return CL_SUCCESS;
}


cl_int pocl_almaif_init(unsigned j, cl_device_id dev, const char *parameters) {

  SETUP_DEVICE_CL_VERSION(dev, 1, 2);
  dev->type = CL_DEVICE_TYPE_CUSTOM;
  dev->long_name = (char *)"memory mapped custom device";
  dev->short_name = "almaif";
  dev->vendor = "pocl";
  dev->version = "1.2";
  dev->extensions = "";
  dev->profile = "FULL_PROFILE";

  dev->profiling_timer_resolution = 1000;
  dev->profile = "EMBEDDED_PROFILE";
  // TODO not sure about these 2
  dev->max_mem_alloc_size = 100 * 1024 * 1024;
  dev->mem_base_addr_align = 16;
  dev->max_constant_buffer_size = 32768;
  dev->local_mem_size = 16384;
  dev->max_work_item_dimensions = 3;
  // kernel param size. this is a bit arbitrary
  dev->max_parameter_size = 64;
  dev->address_bits = 32;

  dev->device_side_printf = 0;

  // This would be more logical as a per builtin kernel value?
  // there is a way to query it: clGetKernelWorkGroupInfo
  /*
   * CL_​KERNEL_​GLOBAL_​WORK_​SIZE
   * query the maximum global size that can be used to execute a kernel
   * (i.e. global_work_size argument to clEnqueueNDRangeKernel) on a custom
   * device
   *
   * CL_​KERNEL_​WORK_​GROUP_​SIZE
   * query the maximum workgroup size that can be used to execute the kernel
   * on a specific device given by device.
   */
  // Either way, the minimum is 3 for a device
  dev->max_work_item_dimensions = 3;
  dev->max_work_group_size = dev->max_work_item_sizes[0] =
      dev->max_work_item_sizes[1] = dev->max_work_item_sizes[2] = 1024;
  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1] =
      dev->max_work_item_sizes[2] = dev->max_work_group_size = 64;
  dev->preferred_wg_size_multiple = 8;

  AlmaifData *D = new AlmaifData;
  D->Available = CL_TRUE;
  dev->available = &(D->Available);
  dev->data = (void *)D;

  char *scanParams;
  if (parameters) {
    // strtok_r() modifies string, copy it to be nice
    scanParams = strdup(parameters);
  } else {
    POCL_MSG_PRINT_ALMAIF("Parameters were not given. Creating emulation "
                          "device with built-in kernels 0,1,2,4,9.\n");
    scanParams = strdup("0xE,0,1,2,4,9");
  }

  char *savePtr;
  char *paramToken = strtok_r(scanParams, ",", &savePtr);
  D->BaseAddress = strtoull(paramToken, NULL, 0);

  std::string supportedList;
  std::string device_init_file = "";
  if (D->BaseAddress != POCL_ALMAIFDEVICE_EMULATION) {
    paramToken = strtok_r(NULL, ",", &savePtr);
    assert(paramToken);
    device_init_file = paramToken;
    POCL_MSG_PRINT_ALMAIF("Enabling device with device init file name %s\n",
                          device_init_file.c_str());
  }

  bool enable_compilation = false;

  // almaif devices are little endian by default, but the emulation device is
  // host dependant
  dev->endian_little = D->BaseAddress == POCL_ALMAIFDEVICE_EMULATION
                           ? !(WORDS_BIGENDIAN)
                           : CL_TRUE;
  if (D->BaseAddress == POCL_ALMAIFDEVICE_EMULATION) {
    dev->long_name = (char *)"almaif emulation device";
  }

  if (!pocl_offline_compile) {

    // Recognize whether we are emulating or not
    if (D->BaseAddress == POCL_ALMAIFDEVICE_EMULATION) {
      D->Dev = new EmulationDevice();
    } else if (D->BaseAddress == POCL_ALMAIFDEVICE_XRT) {
#ifdef HAVE_XRT
      D->Dev = new XilinxXrtDevice(device_init_file, j);
#else
      POCL_ABORT(
          "Almaif: tried enabling XilinxXrtDevice but it's not available\n");
#endif
    } else if (D->BaseAddress == POCL_ALMAIFDEVICE_TTASIM) {
#ifdef TCE_AVAILABLE
      D->Dev = new TTASimDevice(device_init_file);
      enable_compilation = true;
#else
      POCL_ABORT("almaif: Tried enabling TTASim device, but it's not available. "
                 "Did you set ENABLE_TCE=1?\n");
#endif
    } else if (D->BaseAddress == POCL_ALMAIFDEVICE_BITSTREAMDATABASE) {
#ifdef HAVE_DBDEVICE
      D->Dev = new DBDevice(device_init_file);
#else
      POCL_ABORT("Almaif: tried enabling DBDevice but it's not available\n");
#endif
    } else {
      D->Dev = new MMAPDevice(D->BaseAddress, device_init_file);
    }

    if (!(D->Dev->RelativeAddressing)) {
      POCL_LOCK(globalMemIDLock);
      if (isGlobalMemIDSet) {
        dev->global_mem_id = GlobalMemID;
      } else {
        GlobalMemID = dev->dev_id;
        isGlobalMemIDSet = true;
      }
      POCL_UNLOCK(globalMemIDLock);
    }
    dev->global_mem_size = D->Dev->DataMemory->Size();
    if (D->Dev->ExternalMemory != nullptr &&
        D->Dev->ExternalMemory->Size() > D->Dev->DataMemory->Size())
      dev->global_mem_size = D->Dev->ExternalMemory->Size();

  } else {
    POCL_MSG_PRINT_ALMAIF(
        "Starting offline compilation device initialization\n");
  }

  if (D->Dev->isDBDevice()) {
#ifdef HAVE_DBDEVICE
    std::vector<BuiltinKernelId> bik_list =
        ((DBDevice *)(D->Dev))->supportedBuiltinKernels();

    for (const BuiltinKernelId &kernelId : bik_list) {

      bool found = false;
      for (size_t i = 0; i < BIKERNELS; ++i) {
        if (pocl_BIDescriptors[i].KernelId == kernelId) {
          if (supportedList.size() > 0)
            supportedList += ";";
          supportedList += pocl_BIDescriptors[i].name;
          D->SupportedKernels.insert(&pocl_BIDescriptors[i]);
          found = true;
          break;
        }
      }
      if (kernelId == POCL_CDBI_JIT_COMPILER) {
        enable_compilation = true;
      } else if (!found) {
        POCL_ABORT("almaif: Unknown Kernel ID (%i) coming from database\n",
                   kernelId);
      }
    }
#endif
  } else {
    while ((paramToken = strtok_r(NULL, ",", &savePtr))) {
      auto token = strtoul(paramToken, NULL, 0);
      BuiltinKernelId kernelId = static_cast<BuiltinKernelId>(token);

      bool found = false;
      for (size_t i = 0; i < BIKERNELS; ++i) {
        if (pocl_BIDescriptors[i].KernelId == kernelId) {
          if (supportedList.size() > 0)
            supportedList += ";";
          supportedList += pocl_BIDescriptors[i].name;
          D->SupportedKernels.insert(&pocl_BIDescriptors[i]);
          found = true;
          break;
        }
      }
      if (kernelId == POCL_CDBI_JIT_COMPILER) {
        enable_compilation = true;
      } else if (!found) {
        POCL_ABORT("almaif: Unknown Kernel ID (%lu) given\n", token);
      }
    }
  }

  dev->builtin_kernel_list = strdup(supportedList.c_str());
  dev->num_builtin_kernels = D->SupportedKernels.size();
  pocl_setup_builtin_kernels_with_version(dev);
  POCL_MSG_PRINT_ALMAIF(
      "almaif: accelerator at 0x%zx with %zu builtin kernels (%s)\n",
      D->BaseAddress, D->SupportedKernels.size(), dev->builtin_kernel_list);

  free(scanParams);

  if (D->Dev->pipeCount() > 0) {
    dev->pipe_support = CL_TRUE;
    dev->max_pipe_args = ALMAIF_MAX_PIPE_COUNT;
    dev->max_pipe_active_res = 1;
    dev->max_pipe_packet_size = 512;
  }

  if (enable_compilation) {

    dev->compiler_available = CL_TRUE;
    dev->linker_available = CL_TRUE;
    std::string adf_file = device_init_file + ".adf";
    pocl_almaif_compile_init(j, dev, adf_file);

  } else {
    D->compilationData = NULL;
    dev->compiler_available = CL_FALSE;
    dev->linker_available = CL_FALSE;
  }

  dev->device_side_printf = 1;
  dev->printf_buffer_size = PRINTF_BUFFER_SIZE / 4;
  chunk_info_t *chunk = NULL;
  chunk = pocl_alloc_buffer(D->Dev->AllocRegions, dev->printf_buffer_size);
  if (chunk == NULL) {
    POCL_MSG_WARN("Almaif: Can't allocate %lu bytes for printf buffer\n",
                  dev->printf_buffer_size);
    dev->device_side_printf = 0;
  } else {
    POCL_MSG_PRINT_ALMAIF("Allocated printf buffer of size %lu from %lu\n",
                          dev->printf_buffer_size, chunk->start_address);
    D->PrintfBuffer = chunk;

    D->PrintfPosition = pocl_alloc_buffer(D->Dev->AllocRegions, 4);
    if (D->PrintfPosition == NULL) {
      POCL_ABORT("Almaif: Can't allocate 4 bytes for printf index\n");
    }
  }

  POCL_MSG_PRINT_ALMAIF("almaif: mmap done\n");
  if (pocl_offline_compile) {
    std::cout << "Offline compilation device initialized" << std::endl;
    return CL_SUCCESS;
  }
  for (unsigned i = 0; i < (D->Dev->DataMemory->Size() >> 2); i++) {
    //    D->Dev->DataMemory->Write32(4 * i, 0);
  }
  for (unsigned i = 0; i < (D->Dev->CQMemory->Size() >> 2); i++) {
    //    D->Dev->CQMemory->Write32(4 * i, 0);
  }
  // Initialize AQL queue by setting all headers to invalid
  POCL_MSG_PRINT_ALMAIF("Initializing AQL Packet cqmemory size=%zu\n",
                        D->Dev->CQMemory->Size());
  for (uint32_t i = AQL_PACKET_LENGTH; i < D->Dev->CQMemory->Size();
       i += AQL_PACKET_LENGTH) {
    D->Dev->CQMemory->Write16(i, AQL_PACKET_INVALID);
  }
  D->Dev->CQMemory->Write32(ALMAIF_CQ_WRITE, 0);
  D->Dev->CQMemory->Write32(ALMAIF_CQ_READ, 0);

#ifdef ALMAIF_DUMP_MEMORY
  POCL_MSG_PRINT_ALMAIF("INIT MEMORY DUMP\n");
  D->Dev->printMemoryDump();
#endif

  // Lift accelerator reset
  D->Dev->ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_CONTINUE_CMD);
  D->Dev->HwClockStart = pocl_gettimemono_ns();

  D->ReadyList = NULL;
  D->CommandList = NULL;
  POCL_INIT_LOCK(D->CommandListLock);
  POCL_INIT_LOCK(D->AQLQueueLock);

  POCL_LOCK(runningDeviceLock);
  if (runningDeviceCount == 0) {
    runningJoinRequested = false;
    POCL_CREATE_THREAD(runningThread, &runningThreadFunc, NULL);
  }
  runningDeviceCount++;
  POCL_UNLOCK(runningDeviceLock);

  if (D->BaseAddress == POCL_ALMAIFDEVICE_EMULATION) {
    POCL_MSG_PRINT_ALMAIF("Custom emulation device %d initialized \n", j);
  } else {
    POCL_MSG_PRINT_ALMAIF("Custom device %d initialized \n", j);
  }
  return CL_SUCCESS;
}

cl_int pocl_almaif_uninit(unsigned j, cl_device_id device) {
  POCL_MSG_PRINT_ALMAIF("almaif: uninit\n");

  POCL_LOCK(runningDeviceLock);
  runningDeviceCount--;
  if (runningDeviceCount == 0) {
    runningJoinRequested = true;
    POCL_JOIN_THREAD(runningThread);
  }
  POCL_UNLOCK(runningDeviceLock);

  AlmaifData *D = (AlmaifData *)device->data;
  if (D->compilationData != NULL) {
    pocl_almaif_compile_uninit(j, device);
    D->compilationData = NULL;
  }

  delete D->Dev;
  delete D;
  return CL_SUCCESS;
}

unsigned int pocl_almaif_probe(struct pocl_device_ops *ops) {
  int env_count = pocl_device_get_env_count(ops->device_name);

  if (env_count < 0) {
    return 1;
  } else {
    return env_count;
  }
}

char *pocl_almaif_build_hash(cl_device_id /*device*/) {
  char *res = (char *)calloc(1000, sizeof(char));
  snprintf(res, 1000, "almaif-%s", HOST_DEVICE_BUILD_HASH);
  return res;
}

void pocl_almaif_update_event(cl_device_id device, cl_event event) {
  AlmaifData *D = (AlmaifData *)device->data;
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  union {
    struct {
      uint32_t a;
      uint32_t b;
    } u32;
    uint64_t u64;
  } timestamp;

  if ((event->queue->properties & CL_QUEUE_PROFILING_ENABLE) &&
      (event->command_type == CL_COMMAND_NDRANGE_KERNEL)) {

    if (event->status <= CL_COMPLETE) {
      assert(ed);
      size_t commandMetaAddress = ed->chunk->start_address;
      assert(commandMetaAddress);
      commandMetaAddress -= D->Dev->DataMemory->PhysAddress();

      timestamp.u32.a = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, start_timestamp));
      timestamp.u32.b = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, start_timestamp) +
          sizeof(uint32_t));
      if (timestamp.u64 > 0)
        event->time_start = timestamp.u64;

      timestamp.u32.a = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, finish_timestamp));
      timestamp.u32.b = D->Dev->DataMemory->Read32(
          commandMetaAddress + offsetof(CommandMetadata, finish_timestamp) +
          sizeof(uint32_t));
      if (timestamp.u64 > 0)
        event->time_end = timestamp.u64;

      // recalculation of timestamps to host clock
      if (D->Dev->HasHardwareClock && D->Dev->HwClockFrequency > 0) {
        double NsPerClock =
            (double)1000000000.0 / (double)D->Dev->HwClockFrequency;

        double StartNs = (double)event->time_start * NsPerClock;
        event->time_start = D->Dev->HwClockStart + (uint64_t)StartNs;

        double EndNs = (double)event->time_end * NsPerClock;
        event->time_end = D->Dev->HwClockStart + (uint64_t)EndNs;
      }

      if (device->device_side_printf) {
        chunk_info_t *PrintfBufferChunk = (chunk_info_t *)D->PrintfBuffer;
        assert(PrintfBufferChunk);
        chunk_info_t *PrintfPositionChunk = (chunk_info_t *)D->PrintfPosition;
        assert(PrintfPositionChunk);
        unsigned position =
            D->Dev->DataMemory->Read32(PrintfPositionChunk->start_address -
                                       D->Dev->DataMemory->PhysAddress());
        POCL_MSG_PRINT_ALMAIF(
            "Device wrote %u bytes to stdout. Printing them now:\n", position);
        if (position > 0) {
          char *tmp_printf_buf = (char *)malloc(position);
          D->Dev->DataMemory->CopyFromMMAP(
              tmp_printf_buf, PrintfBufferChunk->start_address, position);
          D->Dev->DataMemory->Write32(PrintfPositionChunk->start_address -
                                          D->Dev->DataMemory->PhysAddress(),
                                      0);
          write(STDOUT_FILENO, tmp_printf_buf, position);
          free(tmp_printf_buf);
        }
      }
    }
  }
}

static void scheduleCommands(AlmaifData &D) {

  _cl_command_node *Node;
  // Execute commands from ready list.
  while ((Node = D.ReadyList)) {
    assert(Node->sync.event.event->status == CL_SUBMITTED);

    if (Node->type == CL_COMMAND_NDRANGE_KERNEL) {
      pocl_update_event_running(Node->sync.event.event);

      submit_and_barrier((AlmaifData *)Node->device->data, Node);
      submit_kernel_packet((AlmaifData *)Node->device->data, Node);

      POCL_LOCK(runningLock);
      CDL_DELETE(D.ReadyList, Node);
      DL_PREPEND(runningList, Node);
      POCL_UNLOCK(runningLock);
    } else {
      assert(pocl_command_is_ready(Node->sync.event.event));

      CDL_DELETE(D.ReadyList, Node);
      POCL_UNLOCK(D.CommandListLock);
      pocl_exec_command(Node);
      POCL_LOCK(D.CommandListLock);
    }
  }

  return;
}

bool only_custom_device_events_left(cl_event event) {
  event_node *dep_event = event->wait_list;

  // When the next kernel uses a different jit-compiled program
  // we cannot insert and-barrier, since the host must be able to switch
  // the program in between the kernels.
  // TODO: fix this check to work if the program is the same, or if the program
  // contains both this kernel and dependent kernels.
  // Current solution only adds and-barriers in case the device supports just
  // built-in kernels.
  if (event->queue->device->compiler_available && dep_event) {
    return false;
  }
  while (dep_event) {
    POCL_MSG_PRINT_ALMAIF("Looking at event id=%" PRIu64 "\n",
                         dep_event->event->id);
    const char *dep_dev_name = dep_event->event->queue->device->short_name;
    const char *dev_name = event->queue->device->short_name;
    if ((dep_event->event->command->type != CL_COMMAND_NDRANGE_KERNEL) ||
        strcmp(dep_dev_name, dev_name)) {
      POCL_MSG_PRINT_ALMAIF(
          "Found a dependent non-almaif event, have to wait on this cmd\n");
      return false;
    }
    dep_event = dep_event->next;
  }
  POCL_MSG_PRINT_ALMAIF("No non-custom events left, let this cmd through\n");
  return true;
}

void pocl_almaif_submit(_cl_command_node *Node, cl_command_queue /*CQ*/) {

  Node->node_state = COMMAND_READY;

  struct AlmaifData *D = (AlmaifData *)Node->device->data;
  cl_event E = Node->sync.event.event;

  if (E->data == nullptr) {
    E->data = calloc(1, sizeof(almaif_event_data_t));
    assert(E->data && "event data allocation failed");
    almaif_event_data_t *ed = (almaif_event_data_t *)E->data;
    POCL_INIT_COND(ed->event_cond);
  }

  if ((Node->type == CL_COMMAND_NDRANGE_KERNEL) &&
      only_custom_device_events_left(E)) {
    pocl_update_event_submitted(E);

    POCL_UNLOCK_OBJ(E);
    pocl_update_event_running(E);

    submit_and_barrier(D, Node);
    submit_kernel_packet(D, Node);

    POCL_LOCK(runningLock);
    DL_PREPEND(runningList, Node);
    POCL_UNLOCK(runningLock);
  } else {
    POCL_LOCK(D->CommandListLock);
    pocl_command_push(Node, &D->ReadyList, &D->CommandList);

    POCL_UNLOCK_OBJ(E);
    scheduleCommands(*D);
    POCL_UNLOCK(D->CommandListLock);
  }

  return;
}

void pocl_almaif_join(cl_device_id device, cl_command_queue cq) {

  struct AlmaifData *D = (AlmaifData *)device->data;
  POCL_LOCK(D->CommandListLock);
  while (D->CommandList || D->ReadyList) {
    scheduleCommands(*D);
    POCL_UNLOCK(D->CommandListLock);
    usleep(ALMAIF_DRIVER_SLEEP);
    POCL_LOCK(D->CommandListLock);
  }

  POCL_UNLOCK(D->CommandListLock);

  POCL_LOCK_OBJ(cq);
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  assert(cq_cond);
  while (1) {
    if (cq->command_count == 0) {
      POCL_UNLOCK_OBJ(cq);
      return;
    } else {
      POCL_WAIT_COND(*cq_cond, cq->pocl_lock);
    }
  }
  return;
}

void pocl_almaif_notify(cl_device_id Device, cl_event Event, cl_event Finished) {

  struct AlmaifData &D = *(AlmaifData *)Device->data;

  _cl_command_node *volatile Node = Event->command;

  if (Finished->status < CL_COMPLETE) {
    pocl_update_event_failed(Event);
    return;
  }

  if (Node->node_state != COMMAND_READY)
    {
      POCL_MSG_PRINT_EVENTS (
          "almaif: command related to the notified event %lu not ready\n",
          Event->id);
      return;
    }

  if (Event->command->type != CL_COMMAND_NDRANGE_KERNEL) {
    if (pocl_command_is_ready(Event)) {
      if (Event->status == CL_QUEUED) {
        pocl_update_event_submitted(Event);
        POCL_LOCK(D.CommandListLock);
        CDL_DELETE(D.CommandList, Node);
        CDL_PREPEND(D.ReadyList, Node);

        POCL_UNLOCK_OBJ(Event);
        scheduleCommands(D);
        POCL_LOCK_OBJ(Event);

        POCL_UNLOCK(D.CommandListLock);
      }
    }
  } else {
    if (only_custom_device_events_left(Event)) {
      pocl_update_event_submitted(Event);
      POCL_LOCK(D.CommandListLock);
      CDL_DELETE(D.CommandList, Node);
      CDL_PREPEND(D.ReadyList, Node);

      POCL_UNLOCK_OBJ(Event);
      scheduleCommands(D);
      POCL_LOCK_OBJ(Event);

      POCL_UNLOCK(D.CommandListLock);
    }
  }
}

void scheduleNDRange(AlmaifData *data, _cl_command_node *cmd, size_t arg_size,
                     void *arguments) {
  _cl_command_run *run = &cmd->command.run;
  cl_kernel k = run->kernel;
  cl_program p = k->program;
  cl_event e = cmd->sync.event.event;
  almaif_event_data_t *event_data = (almaif_event_data_t *)e->data;
  int32_t kernelID = -1;
  bool SanitizeKernelName = false;

  for (auto supportedKernel : data->SupportedKernels) {
    if (strcmp(supportedKernel->name, k->name) == 0) {
      kernelID = (int32_t)supportedKernel->KernelId;
      /* builtin kernels that come from tce_kernels.cl need compiling */
      if (p->num_builtin_kernels > 0 && p->source) {
        POCL_MSG_PRINT_ALMAIF(
            "almaif: builtin kernel with source, needs compiling\n");
        kernelID = -1;
        SanitizeKernelName = true;
      }
      break;
    }
  }
#ifdef HAVE_DBDEVICE
  if (data->Dev->isDBDevice()) {
    ((DBDevice *)(data->Dev))
        ->programBIKernelBitstream((BuiltinKernelId)kernelID);
    ((DBDevice *)(data->Dev))
        ->programBIKernelFirmware((BuiltinKernelId)kernelID);
  }
#endif

  if (kernelID == -1) {
    if (data->compilationData == NULL) {
      POCL_ABORT("almaif: scheduled an NDRange with unsupported kernel\n");
    } else {
      POCL_MSG_PRINT_ALMAIF("almaif: compiling kernel\n");
      char *SavedName = nullptr;
      if (SanitizeKernelName)
        pocl_sanitize_builtin_kernel_name(k, &SavedName);

      pocl_almaif_compile_kernel(cmd, k, cmd->device, 1);

      if (SanitizeKernelName)
        pocl_restore_builtin_kernel_name(k, SavedName);
    }
  }

  // Additional space for a signal
  size_t extraAlloc = sizeof(struct CommandMetadata);
  chunk_info_t *chunk = pocl_alloc_buffer_from_region(data->Dev->AllocRegions,
                                                      arg_size + extraAlloc);
  assert(chunk && "Failed to allocate signal/argument buffer");

  POCL_MSG_PRINT_ALMAIF("almaif: allocated 0x%zx bytes for signal/arguments "
                       "from 0x%zx\n",
                       arg_size + extraAlloc, chunk->start_address);
  assert(event_data);
  assert(event_data->chunk == NULL);
  event_data->chunk = chunk;

  size_t commandMetaAddress = chunk->start_address;
  size_t signalAddress =
      commandMetaAddress + offsetof(CommandMetadata, completion_signal);
  size_t argsAddress = chunk->start_address + sizeof(struct CommandMetadata);
  POCL_MSG_PRINT_ALMAIF("Signal address=0x%zx\n", signalAddress);
  // clear the timestamps and initial signal value
  for (unsigned offset = 0; offset < sizeof(CommandMetadata); offset += 4)
    data->Dev->DataMemory->Write32(
        commandMetaAddress - data->Dev->DataMemory->PhysAddress() + offset, 0);
  if (cmd->device->device_side_printf) {
    data->Dev->DataMemory->Write32(
        commandMetaAddress - data->Dev->DataMemory->PhysAddress() +
            offsetof(CommandMetadata, reserved0),
        ((chunk_info_t *)data->PrintfBuffer)->start_address);
    data->Dev->DataMemory->Write32(commandMetaAddress -
                                       data->Dev->DataMemory->PhysAddress() +
                                       offsetof(CommandMetadata, reserved1),
                                   cmd->device->printf_buffer_size);

    data->Dev->DataMemory->Write32(
        commandMetaAddress - data->Dev->DataMemory->PhysAddress() +
            offsetof(CommandMetadata, reserved1) + 4,
        ((chunk_info_t *)data->PrintfPosition)->start_address);
  }

  // Set arguments
  data->Dev->DataMemory->CopyToMMAP(argsAddress, arguments, arg_size);

  struct AQLDispatchPacket packet = {};

  packet.header = AQL_PACKET_INVALID;
  packet.dimensions = run->pc.work_dim; // number of dimensions

  packet.workgroup_size_x = run->pc.local_size[0];
  packet.workgroup_size_y = run->pc.local_size[1];
  packet.workgroup_size_z = run->pc.local_size[2];

  pocl_context32 pc;

  if (kernelID != -1) {
    packet.grid_size_x = run->pc.local_size[0] * run->pc.num_groups[0];
    packet.grid_size_y = run->pc.local_size[1] * run->pc.num_groups[1];
    packet.grid_size_z = run->pc.local_size[2] * run->pc.num_groups[2];
    packet.kernel_object = kernelID;
  } else {

    // Compilation needs pocl_context struct, create it, copy it to device and
    // pass the pointer to it in the 'reserved' slot of AQL kernel dispatch
    // packet.
    pc.work_dim = run->pc.work_dim;
    pc.local_size[0] = run->pc.local_size[0];
    pc.local_size[1] = run->pc.local_size[1];
    pc.local_size[2] = run->pc.local_size[2];
    pc.num_groups[0] = run->pc.num_groups[0];
    pc.num_groups[1] = run->pc.num_groups[1];
    pc.num_groups[2] = run->pc.num_groups[2];
    pc.global_offset[0] = run->pc.global_offset[0];
    pc.global_offset[1] = run->pc.global_offset[1];
    pc.global_offset[2] = run->pc.global_offset[2];
    pc.global_var_buffer = 0;

    if (cmd->device->device_side_printf) {
      pc.printf_buffer = ((chunk_info_t *)data->PrintfBuffer)->start_address;
      pc.printf_buffer_capacity = cmd->device->printf_buffer_size;
      assert(pc.printf_buffer_capacity);

      pc.printf_buffer_position =
          ((chunk_info_t *)data->PrintfPosition)->start_address;
      POCL_MSG_PRINT_ALMAIF(
          "Device side printf buffer=%d, position: %d and capacity %d \n",
          pc.printf_buffer, pc.printf_buffer_position,
          pc.printf_buffer_capacity);

      data->Dev->DataMemory->Write32(
          pc.printf_buffer_position - data->Dev->DataMemory->PhysAddress(), 0);
    }

    size_t pc_start_addr = data->compilationData->pocl_context->start_address;
    data->Dev->DataMemory->CopyToMMAP(pc_start_addr, &pc,
                                      sizeof(pocl_context32));

    if (data->Dev->RelativeAddressing) {
      pc_start_addr -= data->Dev->DataMemory->PhysAddress();
    }

    packet.reserved = pc_start_addr;

    almaif_kernel_data_t *kd =
        (almaif_kernel_data_t *)run->kernel->data[cmd->program_device_i];
    packet.kernel_object = kd->kernel_address;

    POCL_MSG_PRINT_ALMAIF("Kernel addresss=0x%" PRIu32 "\n", kd->kernel_address);
  }

  if (data->Dev->RelativeAddressing) {
    packet.kernarg_address = argsAddress - data->Dev->DataMemory->PhysAddress();
    packet.command_meta_address =
        commandMetaAddress - data->Dev->DataMemory->PhysAddress();
  } else {
    packet.kernarg_address = argsAddress;
    packet.command_meta_address = commandMetaAddress;
  }

  POCL_MSG_PRINT_ALMAIF("ArgsAddress=0x%" PRIx64 " CommandMetaAddress=0x%" PRIx64
                       " \n",
                       packet.kernarg_address, packet.command_meta_address);

  POCL_LOCK(data->AQLQueueLock);
  uint32_t queue_length = data->Dev->CQMemory->Size() / AQL_PACKET_LENGTH - 1;

  uint32_t write_iter = data->Dev->CQMemory->Read32(ALMAIF_CQ_WRITE);
  uint32_t read_iter = data->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
  while (write_iter >= read_iter + queue_length) {
    POCL_MSG_PRINT_ALMAIF("write_iter=%u, read_iter=%u length=%u", write_iter,
                          read_iter, queue_length);
    usleep(ALMAIF_DRIVER_SLEEP);
    read_iter = data->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
#ifdef ALMAIF_DUMP_MEMORY
    POCL_MSG_PRINT_ALMAIF("WAITING FOR CQMEMORY TO EMPTY DUMP\n");
    data->Dev->printMemoryDump();
#endif
  }
  uint32_t packet_loc =
      (write_iter % queue_length) * AQL_PACKET_LENGTH + AQL_PACKET_LENGTH;
  data->Dev->CQMemory->CopyToMMAP(
      packet_loc + data->Dev->CQMemory->PhysAddress(), &packet, 64);

#ifdef ALMAIF_DUMP_MEMORY
  POCL_MSG_PRINT_ALMAIF("PRELAUNCH MEMORY DUMP\n");
  data->Dev->printMemoryDump();
#endif

  // finally, set header as not-invalid
  data->Dev->CQMemory->Write16(packet_loc, (1 << AQL_PACKET_KERNEL_DISPATCH) |
                                               AQL_PACKET_BARRIER);

  POCL_MSG_PRINT_ALMAIF(
      "almaif: Handed off a packet for execution, write iter=%u\n", write_iter);
  // Increment queue index
  data->Dev->CQMemory->Write32(ALMAIF_CQ_WRITE, write_iter + 1);

  POCL_UNLOCK(data->AQLQueueLock);

  if (kernelID == -1 && data->compilationData &&
      data->compilationData->produce_standalone_program) {
    data->compilationData->produce_standalone_program(data, cmd, &pc, arg_size,
                                                      arguments);
  }
}

bool isEventDone(AlmaifData *data, cl_event event) {

  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  if (ed->chunk->start_address == 0)
    return false;

  size_t commandMetaAddress = ed->chunk->start_address;
  assert(commandMetaAddress);
  size_t signalAddress =
      commandMetaAddress + offsetof(CommandMetadata, completion_signal);
  signalAddress -= data->Dev->DataMemory->PhysAddress();

  uint32_t status = data->Dev->DataMemory->Read32(signalAddress);

  if (status == 1) {
    POCL_MSG_PRINT_ALMAIF("Event %" PRIu64
                         " done, completion signal address=%zx, value=%u\n",
                         event->id, signalAddress, status);
  }

  return (status == 1);
}

void pocl_almaif_wait_event(cl_device_id device, cl_event event) {
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;

  POCL_LOCK_OBJ(event);
  while (event->status > CL_COMPLETE) {
    POCL_WAIT_COND(ed->event_cond, event->pocl_lock);
  }
  POCL_UNLOCK_OBJ(event);
}

void pocl_almaif_notify_cmdq_finished(cl_command_queue cq) {
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  PTHREAD_CHECK(pthread_cond_broadcast(cq_cond));
}

void pocl_almaif_notify_event_finished(cl_event event) {
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  POCL_BROADCAST_COND(ed->event_cond);

  /* this is a hack required b/c pocld does not release events,
   * the "pocl_almaif_free_event_data" is not called, and because
   * almaif allocates memory from device globalmem for signals,
   * the device eventually runs out of memory. */
  if (event->command_type == CL_COMMAND_NDRANGE_KERNEL && ed->chunk != NULL) {
    pocl_free_chunk((chunk_info_t *)ed->chunk);
    ed->chunk = NULL;
  }
}

int pocl_almaif_init_queue(cl_device_id device, cl_command_queue queue) {
  queue->data = malloc(sizeof(pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  POCL_INIT_COND(*cond);
  return CL_SUCCESS;
}

int pocl_almaif_free_queue(cl_device_id device, cl_command_queue queue) {
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  POCL_DESTROY_COND(*cond);
  POCL_MEM_FREE(queue->data);
  return CL_SUCCESS;
}

void submit_and_barrier(AlmaifData *D, _cl_command_node *cmd) {

  event_node *dep_event = cmd->sync.event.event->wait_list;
  if (dep_event == NULL) {
    POCL_MSG_PRINT_ALMAIF("Almaif: no events to wait for\n");
    return;
  }

  bool all_done = false;
  while (!all_done) {
    struct AQLAndPacket packet = {};
    memset(&packet, 0, sizeof(AQLAndPacket));
    packet.header = AQL_PACKET_INVALID;
    int i;
    for (i = 0; i < AQL_MAX_SIGNAL_COUNT; i++) {
      almaif_event_data_t *dep_ed = (almaif_event_data_t *)dep_event->event->data;
      assert(dep_ed);
      if (dep_ed->chunk) {
        packet.dep_signals[i] = dep_ed->chunk->start_address;
        POCL_MSG_PRINT_ALMAIF(
            "Creating AND barrier depending on signal id=%" PRIu64
            " at address %" PRIu64 " \n",
            dep_event->event->id, packet.dep_signals[i]);
      }
      dep_event = dep_event->next;
      if (dep_event == NULL) {
        all_done = true;
        break;
      }
    }
    packet.signal_count = i + 1;

    POCL_LOCK(D->AQLQueueLock);
    uint32_t queue_length = D->Dev->CQMemory->Size() / AQL_PACKET_LENGTH - 1;

    uint32_t write_iter = D->Dev->CQMemory->Read32(ALMAIF_CQ_WRITE);
    uint32_t read_iter = D->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
    while (write_iter >= read_iter + queue_length) {
      POCL_MSG_PRINT_ALMAIF("write_iter=%u, read_iter=%u length=%u", write_iter,
                            read_iter, queue_length);
      read_iter = D->Dev->CQMemory->Read32(ALMAIF_CQ_READ);
      usleep(ALMAIF_DRIVER_SLEEP);
    }
    uint32_t packet_loc =
        (write_iter % queue_length) * AQL_PACKET_LENGTH + AQL_PACKET_LENGTH;
    D->Dev->CQMemory->CopyToMMAP(packet_loc + D->Dev->CQMemory->PhysAddress(),
                                 &packet, 64);

    D->Dev->CQMemory->Write16(packet_loc, (1 << AQL_PACKET_BARRIER_AND) |
                                              AQL_PACKET_BARRIER);

    POCL_MSG_PRINT_ALMAIF("almaif: Handed off and barrier, write iter=%u\n",
                         write_iter);
    // Increment queue index
    D->Dev->CQMemory->Write32(ALMAIF_CQ_WRITE, write_iter + 1);

    POCL_UNLOCK(D->AQLQueueLock);
  }
}

void pocl_almaif_run(void *data, _cl_command_node *cmd) {}

void submit_kernel_packet(AlmaifData *D, _cl_command_node *cmd) {
  struct pocl_argument *al;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0 || pc->num_groups[2] == 0)
    return;

  // First pass to figure out total argument size
  size_t arg_size = 0;
  for (i = 0; i < meta->num_args; ++i) {
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      arg_size += D->Dev->PointerSize;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_PIPE) {
      arg_size += 4;
    } else {
      al = &(cmd->command.run.arguments[i]);
      arg_size += al->size;
    }
  }
  void *arguments = malloc(arg_size);
  char *current_arg = (char *)arguments;
  /* TODO: Refactor this to a helper function (the argbuffer ABI). */
  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
  for (i = 0; i < meta->num_args; ++i) {
    al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i]))
      // No kernels with local args at the moment, should not end up here
      POCL_ABORT_UNIMPLEMENTED("almaif: local arguments");
    else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      /* It's legal to pass a NULL pointer to clSetKernelArguments. In
         that case we must pass the same NULL forward to the kernel.
         Otherwise, the user must have created a buffer with per device
         pointers stored in the cl_mem. */
      if (al->value == NULL) {
        *(size_t *)current_arg = 0;
      } else {
        // almaif doesn't support SVM pointers
        assert(al->is_raw_ptr == 0);
        cl_mem m = (*(cl_mem *)(al->value));
        size_t buffer = D->Dev->pointerDeviceOffset(
            &(m->device_ptrs[cmd->device->global_mem_id]));
        buffer += al->offset;
        if (D->Dev->RelativeAddressing) {
          if (D->Dev->DataMemory->isInRange(buffer)) {
            buffer -= D->Dev->DataMemory->PhysAddress();
          } else if (D->Dev->ExternalMemory->isInRange(buffer)) {
            buffer -= D->Dev->ExternalMemory->PhysAddress();
          } else {
            POCL_ABORT("almaif: buffer outside of memory");
          }
        }
        *(size_t *)current_arg = buffer;
      }
      current_arg += D->Dev->PointerSize;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      POCL_ABORT_UNIMPLEMENTED("almaif: image arguments");
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_ABORT_UNIMPLEMENTED("almaif: sampler arguments");
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_PIPE) {
      cl_mem m = (*(cl_mem *)(al->value));
      int *pipe_id_ptr =
          (int *)(m->device_ptrs[cmd->device->global_mem_id].mem_ptr);
      int pipe_id = *pipe_id_ptr;
      *(int *)current_arg = pipe_id;
      POCL_MSG_PRINT_ALMAIF("Setting pipe argument %d to id: %i\n", i, pipe_id);
      current_arg += 4;
    } else {
      memcpy(current_arg, al->value, al->size);
      current_arg += al->size;
    }
  }

  scheduleNDRange(D, cmd, arg_size, arguments);

  POCL_MEM_FREE(cmd->command.run.device_data);
  free(arguments);
}

void pocl_almaif_free_event_data(cl_event event) {
  almaif_event_data_t *ed = (almaif_event_data_t *)event->data;
  if (ed) {
    if (ed->chunk != NULL) {
      pocl_free_chunk((chunk_info_t *)ed->chunk);
    }
    POCL_DESTROY_COND(ed->event_cond);
    POCL_MEM_FREE(event->data);
  }
  event->data = NULL;
}

void *runningThreadFunc(void *) {
  int counter = 0;
  while (!runningJoinRequested) {
    POCL_LOCK(runningLock);
    if (runningList) {
      _cl_command_node *Node = NULL;
      _cl_command_node *tmp = NULL;
      DL_FOREACH_SAFE(runningList, Node, tmp) {
        AlmaifData *AD = (AlmaifData *)Node->device->data;
        if (isEventDone(AD, Node->sync.event.event)) {
          DL_DELETE(runningList, Node);
          cl_event E = Node->sync.event.event;
#ifdef ALMAIF_DUMP_MEMORY
          POCL_MSG_PRINT_ALMAIF("FINAL MEMORY DUMP\n");
          AD->Dev->printMemoryDump();
#endif
          POCL_UNLOCK(runningLock);
          POCL_UPDATE_EVENT_COMPLETE_MSG(E, "Almaif, asynchronous NDRange    ");
          POCL_LOCK(runningLock);
        }

#ifdef ALMAIF_DUMP_MEMORY
        if ((counter % 3) == 0) {
          if (Node->device->device_side_printf) {
            chunk_info_t *PrintfBufferChunk = (chunk_info_t *)AD->PrintfBuffer;
            assert(PrintfBufferChunk);
            chunk_info_t *PrintfPositionChunk =
                (chunk_info_t *)AD->PrintfPosition;
            assert(PrintfPositionChunk);
            unsigned position =
                AD->Dev->DataMemory->Read32(PrintfPositionChunk->start_address -
                                            AD->Dev->DataMemory->PhysAddress());
            POCL_MSG_PRINT_ALMAIF(
                "Device wrote %u bytes to stdout. Printing them now:\n",
                position);
            if (position > 0) {
              char *tmp_printf_buf = (char *)malloc(position);
              AD->Dev->DataMemory->CopyFromMMAP(
                  tmp_printf_buf, PrintfBufferChunk->start_address, position);
              write(STDOUT_FILENO, tmp_printf_buf, position);
              free(tmp_printf_buf);
            }
          }
        } else {
          uint32_t pc = AD->Dev->ControlMemory->Read32(ALMAIF_STATUS_REG_PC);
          uint64_t cc =
              AD->Dev->ControlMemory->Read64(ALMAIF_STATUS_REG_CC_LOW);
          uint64_t sc =
              AD->Dev->ControlMemory->Read64(ALMAIF_STATUS_REG_SC_LOW);
          POCL_MSG_PRINT_ALMAIF(
              "PC:%" PRId32 " CC:%" PRId64 " SC:%" PRId64 "\n", pc, cc, sc);

          POCL_MSG_PRINT_ALMAIF("RUNNING MEMORY DUMP\n");
          AD->Dev->printMemoryDump();
        }
#endif
        counter++;
      }
    }
    POCL_UNLOCK(runningLock);
    usleep(ALMAIF_DRIVER_SLEEP);
  }
  return NULL;
}

void pocl_almaif_copy_rect(void *data, pocl_mem_identifier *dst_mem_id,
                          cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                          cl_mem src_buf,
                          const size_t *__restrict__ const dst_origin,
                          const size_t *__restrict__ const src_origin,
                          const size_t *__restrict__ const region,
                          size_t const dst_row_pitch,
                          size_t const dst_slice_pitch,
                          size_t const src_row_pitch,
                          size_t const src_slice_pitch) {
  AlmaifData *d = (AlmaifData *)data;

  size_t src_offset = src_origin[0] + src_row_pitch * src_origin[1] +
                      src_slice_pitch * src_origin[2];
  size_t dst_offset = dst_origin[0] + dst_row_pitch * dst_origin[1] +
                      dst_slice_pitch * dst_origin[2];

  size_t j, k, i;

  /* TODO: handle overlaping regions */

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      for (i = 0; i < region[0]; i++) {
        char val;
        d->Dev->readDataFromDevice(&val, src_mem_id, 1,
                                   src_offset + src_row_pitch * j +
                                       src_slice_pitch * k + i);
        d->Dev->writeDataToDevice(dst_mem_id, &val, 1,
                                  dst_offset + dst_row_pitch * j +
                                      dst_slice_pitch * k + i);
      }
}

void pocl_almaif_write_rect(void *data, const void *__restrict__ src_host_ptr,
                           pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                           const size_t *__restrict__ const buffer_origin,
                           const size_t *__restrict__ const host_origin,
                           const size_t *__restrict__ const region,
                           size_t const buffer_row_pitch,
                           size_t const buffer_slice_pitch,
                           size_t const host_row_pitch,
                           size_t const host_slice_pitch) {
  AlmaifData *d = (AlmaifData *)data;
  size_t adjusted_dst_offset = buffer_origin[0] +
                               buffer_row_pitch * buffer_origin[1] +
                               buffer_slice_pitch * buffer_origin[2];

  char const *__restrict__ const adjusted_host_ptr =
      (char const *)src_host_ptr + host_origin[0] +
      host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];

  size_t j, k;

  /* TODO: handle overlapping regions */

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j) {
      size_t s_offset = host_row_pitch * j + host_slice_pitch * k;

      size_t d_offset = buffer_row_pitch * j + buffer_slice_pitch * k;

      d->Dev->writeDataToDevice(dst_mem_id, adjusted_host_ptr + s_offset,
                                region[0], adjusted_dst_offset + d_offset);
    }
}

void pocl_almaif_read_rect(void *data, void *__restrict__ dst_host_ptr,
                          pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                          const size_t *__restrict__ const buffer_origin,
                          const size_t *__restrict__ const host_origin,
                          const size_t *__restrict__ const region,
                          size_t const buffer_row_pitch,
                          size_t const buffer_slice_pitch,
                          size_t const host_row_pitch,
                          size_t const host_slice_pitch) {
  AlmaifData *d = (AlmaifData *)data;
  size_t adjusted_src_offset = buffer_origin[0] +
                               buffer_row_pitch * buffer_origin[1] +
                               buffer_slice_pitch * buffer_origin[2];

  char *__restrict__ const adjusted_host_ptr =
      (char *)dst_host_ptr + host_origin[0] + host_row_pitch * host_origin[1] +
      host_slice_pitch * host_origin[2];

  size_t j, k;

  /* TODO: handle overlaping regions */

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j) {
      size_t d_offset = host_row_pitch * j + host_slice_pitch * k;
      size_t s_offset = buffer_row_pitch * j + buffer_slice_pitch * k;
      d->Dev->readDataFromDevice(adjusted_host_ptr + d_offset, src_mem_id,
                                 region[0], adjusted_src_offset + s_offset);
    }
}
