/// level0-driver.cc - driver for LevelZero Compute API devices.
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

#include "level0-driver.hh"

#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_llvm.h"
#include "pocl_spir.h"
#include "pocl_timing.h"
#include "pocl_util.h"

#include "imagefill.h"
#include "memfill.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

// TODO: do we need to use Barriers, if we're using immediate
// cmdlist in synchronous mode
//#define LEVEL0_IMMEDIATE_CMDLIST

using namespace pocl;

static void pocl_level0_abort_on_ze_error(ze_result_t status, unsigned line,
                                          const char *func, const char *code) {
  const char *str = code;
  if (status != ZE_RESULT_SUCCESS) {
    // TODO convert level0 errors to strings
    POCL_MSG_ERR("Error %0x from LevelZero Runtime call:\n", (int)status);
    POCL_ABORT("Code:\n%s\n", str);
  }
}

#define LEVEL0_CHECK_ABORT(code)                                               \
  pocl_level0_abort_on_ze_error(code, __LINE__, __FUNCTION__, #code)

void Level0Queue::runThread() {

  bool ShouldExit = false;
  _cl_command_node *Command = nullptr;
  do {
    BatchType WorkBatch;
    ShouldExit = WorkHandler->getWorkOrWait(&Command, WorkBatch);
    if (Command != nullptr) {
      assert(pocl_command_is_ready(Command->sync.event.event));
      assert(Command->sync.event.event->status == CL_SUBMITTED);
      execCommand(Command);
    }
    if (!WorkBatch.empty()) {
      execCommandBatch(WorkBatch);
    }
  } while (!ShouldExit);
}

void Level0Queue::appendEventToList(_cl_command_node *Cmd, const char **Msg) {
  cl_event event = Cmd->sync.event.event;
  cl_device_id dev = Cmd->device;
  _cl_command_t *cmd = &Cmd->command;
  cl_mem mem = nullptr;
  if (event->num_buffers > 0) {
    mem = event->mem_objs[0];
    assert(mem);
  }

  switch (Cmd->type) {
  case CL_COMMAND_READ_BUFFER:
    read(cmd->read.dst_host_ptr, cmd->read.src_mem_id, mem, cmd->read.offset,
         cmd->read.size);
    *Msg = "Event Read Buffer           ";
    break;

  case CL_COMMAND_WRITE_BUFFER:
    write(cmd->write.src_host_ptr, cmd->write.dst_mem_id, mem,
          cmd->write.offset, cmd->write.size);
    syncUseMemHostPtr(cmd->write.dst_mem_id, mem, cmd->write.offset,
                      cmd->write.size);
    *Msg = "Event Write Buffer          ";
    break;

  case CL_COMMAND_COPY_BUFFER:
    copy(cmd->copy.dst_mem_id, cmd->copy.dst, cmd->copy.src_mem_id,
         cmd->copy.src, cmd->copy.dst_offset, cmd->copy.src_offset,
         cmd->copy.size);
    syncUseMemHostPtr(cmd->copy.dst_mem_id, cmd->copy.dst, cmd->copy.dst_offset,
                      cmd->copy.size);
    *Msg = "Event Copy Buffer           ";
    break;

  case CL_COMMAND_FILL_BUFFER:
    memFill(cmd->memfill.dst_mem_id, mem, cmd->memfill.size,
            cmd->memfill.offset, cmd->memfill.pattern,
            cmd->memfill.pattern_size);
    syncUseMemHostPtr(cmd->memfill.dst_mem_id, mem, cmd->memfill.offset,
                      cmd->memfill.size);
    *Msg = "Event Fill Buffer           ";
    break;

  case CL_COMMAND_READ_BUFFER_RECT:
    readRect(cmd->read_rect.dst_host_ptr, cmd->read_rect.src_mem_id, mem,
             cmd->read_rect.buffer_origin, cmd->read_rect.host_origin,
             cmd->read_rect.region, cmd->read_rect.buffer_row_pitch,
             cmd->read_rect.buffer_slice_pitch, cmd->read_rect.host_row_pitch,
             cmd->read_rect.host_slice_pitch);
    *Msg = "Event Read Buffer Rect      ";
    break;

  case CL_COMMAND_COPY_BUFFER_RECT:
    copyRect(cmd->copy_rect.dst_mem_id, cmd->copy_rect.dst,
             cmd->copy_rect.src_mem_id, cmd->copy_rect.src,
             cmd->copy_rect.dst_origin, cmd->copy_rect.src_origin,
             cmd->copy_rect.region, cmd->copy_rect.dst_row_pitch,
             cmd->copy_rect.dst_slice_pitch, cmd->copy_rect.src_row_pitch,
             cmd->copy_rect.src_slice_pitch);
    syncUseMemHostPtr(cmd->copy_rect.dst_mem_id, cmd->copy_rect.dst,
                      cmd->copy_rect.dst_origin, cmd->copy_rect.region,
                      cmd->copy_rect.dst_row_pitch,
                      cmd->copy_rect.dst_slice_pitch);
    *Msg = "Event Copy Buffer Rect      ";
    break;

  case CL_COMMAND_WRITE_BUFFER_RECT:
    writeRect(cmd->write_rect.src_host_ptr, cmd->write_rect.dst_mem_id, mem,
              cmd->write_rect.buffer_origin, cmd->write_rect.host_origin,
              cmd->write_rect.region, cmd->write_rect.buffer_row_pitch,
              cmd->write_rect.buffer_slice_pitch,
              cmd->write_rect.host_row_pitch, cmd->write_rect.host_slice_pitch);
    syncUseMemHostPtr(cmd->write_rect.dst_mem_id, mem,
                      cmd->write_rect.buffer_origin, cmd->write_rect.region,
                      cmd->write_rect.buffer_row_pitch,
                      cmd->write_rect.buffer_slice_pitch);
    *Msg = "Event Write Buffer Rect     ";
    break;

  case CL_COMMAND_MIGRATE_MEM_OBJECTS:
    switch (cmd->migrate.type) {
    case ENQUEUE_MIGRATE_TYPE_D2H: {
      if (mem->is_image != 0u) {
        size_t region[3] = {mem->image_width, mem->image_height,
                            mem->image_depth};
        if (region[2] == 0) {
          region[2] = 1;
        }
        if (region[1] == 0) {
          region[1] = 1;
        }
        size_t origin[3] = {0, 0, 0};
        readImageRect(mem, cmd->migrate.mem_id, mem->mem_host_ptr, nullptr,
                      origin, region, 0, 0, 0);
      } else {
        read(mem->mem_host_ptr, cmd->migrate.mem_id, mem, 0, mem->size);
      }
      break;
    }
    case ENQUEUE_MIGRATE_TYPE_H2D: {
      if (mem->is_image != 0u) {
        size_t region[3] = {mem->image_width, mem->image_height,
                            mem->image_depth};
        if (region[2] == 0) {
          region[2] = 1;
        }
        if (region[1] == 0) {
          region[1] = 1;
        }
        size_t origin[3] = {0, 0, 0};
        writeImageRect(mem, cmd->migrate.mem_id, mem->mem_host_ptr, nullptr,
                       origin, region, 0, 0, 0);
      } else {
        write(mem->mem_host_ptr, cmd->migrate.mem_id, mem, 0, mem->size);
      }
      break;
    }
    case ENQUEUE_MIGRATE_TYPE_D2D: {
      assert(dev->ops->can_migrate_d2d);
      assert(dev->ops->migrate_d2d);
      dev->ops->migrate_d2d(cmd->migrate.src_device, dev, mem,
                            cmd->migrate.src_id, cmd->migrate.dst_id);
      break;
    }
    case ENQUEUE_MIGRATE_TYPE_NOP: {
      break;
    }
    }
    // TODO sync USE_HOST_PTR
    *Msg = "Event Migrate Buffer(s)     ";
    break;

  case CL_COMMAND_MAP_BUFFER:
    mapMem(cmd->map.mem_id, mem, cmd->map.mapping);
    *Msg = "Event Map Buffer            ";
    break;

  case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    readImageRect(cmd->read_image.src, cmd->read_image.src_mem_id, NULL,
                  cmd->read_image.dst_mem_id, cmd->read_image.origin,
                  cmd->read_image.region, cmd->read_image.dst_row_pitch,
                  cmd->read_image.dst_slice_pitch, cmd->read_image.dst_offset);
    *Msg = "Event CopyImageToBuffer       ";
    break;

  case CL_COMMAND_READ_IMAGE:
    readImageRect(cmd->read_image.src, cmd->read_image.src_mem_id,
                  cmd->read_image.dst_host_ptr, NULL, cmd->read_image.origin,
                  cmd->read_image.region, cmd->read_image.dst_row_pitch,
                  cmd->read_image.dst_slice_pitch, cmd->read_image.dst_offset);
    *Msg = "Event Read Image            ";
    break;

  case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    writeImageRect(cmd->write_image.dst, cmd->write_image.dst_mem_id, NULL,
                   cmd->write_image.src_mem_id, cmd->write_image.origin,
                   cmd->write_image.region, cmd->write_image.src_row_pitch,
                   cmd->write_image.src_slice_pitch,
                   cmd->write_image.src_offset);
    *Msg = "Event CopyBufferToImage       ";
    break;

  case CL_COMMAND_WRITE_IMAGE:
    writeImageRect(cmd->write_image.dst, cmd->write_image.dst_mem_id,
                   cmd->write_image.src_host_ptr, NULL, cmd->write_image.origin,
                   cmd->write_image.region, cmd->write_image.src_row_pitch,
                   cmd->write_image.src_slice_pitch,
                   cmd->write_image.src_offset);
    *Msg = "Event Write Image           ";
    break;

  case CL_COMMAND_COPY_IMAGE:
    copyImageRect(cmd->copy_image.src, cmd->copy_image.dst,
                  cmd->copy_image.src_mem_id, cmd->copy_image.dst_mem_id,
                  cmd->copy_image.src_origin, cmd->copy_image.dst_origin,
                  cmd->copy_image.region);
    *Msg = "Event Copy Image            ";
    break;

  case CL_COMMAND_FILL_IMAGE:
    fillImage(mem, cmd->fill_image.mem_id, cmd->fill_image.origin,
              cmd->fill_image.region, cmd->fill_image.orig_pixel,
              cmd->fill_image.fill_pixel, cmd->fill_image.pixel_size);
    *Msg = "Event Fill Image            ";
    break;

  case CL_COMMAND_MAP_IMAGE:
    mapImage(cmd->map.mem_id, mem, cmd->map.mapping);
    *Msg = "Event Map Image             ";
    break;

  case CL_COMMAND_UNMAP_MEM_OBJECT:
    if (mem->is_image == CL_FALSE || IS_IMAGE1D_BUFFER(mem)) {
      unmapMem(cmd->unmap.mem_id, mem, cmd->unmap.mapping);
      if (cmd->unmap.mapping->map_flags & CL_MAP_WRITE) {
        syncUseMemHostPtr(cmd->unmap.mem_id, mem, cmd->unmap.mapping->offset,
                          cmd->unmap.mapping->size);
      }
    } else {
      unmapImage(cmd->unmap.mem_id, mem, cmd->unmap.mapping);
    }
    *Msg = "Unmap Mem obj         ";
    break;

  case CL_COMMAND_NDRANGE_KERNEL:
    run(Cmd);
    // synchronize content of writable USE_HOST_PTR buffers with the host
    if (event->num_buffers != 0u) {
      for (size_t i = 0; i < event->num_buffers; ++i) {
        mem = event->mem_objs[i];
        if ((mem->flags & CL_MEM_READ_ONLY) != 0u) {
          continue;
        }
        if ((mem->flags & CL_MEM_HOST_NO_ACCESS) != 0u) {
          continue;
        }
        pocl_mem_identifier *mem_id = &mem->device_ptrs[dev->global_mem_id];
        syncUseMemHostPtr(mem_id, mem, 0, mem->size);
      }
    }
    *Msg = "Event Enqueue NDRange       ";
    break;

  case CL_COMMAND_BARRIER:
  case CL_COMMAND_MARKER:
    *Msg = "Event Marker                ";
    break;

  // SVM commands
  case CL_COMMAND_SVM_FREE:
    if (cmd->svm_free.pfn_free_func != nullptr) {
      cmd->svm_free.pfn_free_func(
          cmd->svm_free.queue, cmd->svm_free.num_svm_pointers,
          cmd->svm_free.svm_pointers, cmd->svm_free.data);
    } else {
      for (unsigned i = 0; i < cmd->svm_free.num_svm_pointers; i++) {
        void *ptr = cmd->svm_free.svm_pointers[i];
        POCL_LOCK_OBJ(event->context);
        pocl_raw_ptr *tmp = nullptr;
        pocl_raw_ptr *item = nullptr;
        DL_FOREACH_SAFE(event->context->raw_ptrs, item, tmp) {
          if (item->vm_ptr == ptr) {
            DL_DELETE(event->context->raw_ptrs, item);
            break;
          }
        }
        POCL_UNLOCK_OBJ(event->context);
        assert(item);
        POCL_MEM_FREE(item);
        POname(clReleaseContext)(event->context);
        dev->ops->svm_free(dev, ptr);
      }
    }
    *Msg = "Event SVM Free              ";
    break;

  case CL_COMMAND_SVM_MAP:
    svmMap(cmd->svm_map.svm_ptr);
    *Msg = "Event SVM Map              ";
    break;

  case CL_COMMAND_SVM_UNMAP:
    svmUnmap(cmd->svm_unmap.svm_ptr);
    *Msg = "Event SVM Unmap             ";
    break;

  case CL_COMMAND_SVM_MEMCPY:
  case CL_COMMAND_MEMCPY_INTEL:
    svmCopy(cmd->svm_memcpy.dst, cmd->svm_memcpy.src, cmd->svm_memcpy.size);
    *Msg = "Event SVM Memcpy            ";
    break;

  case CL_COMMAND_SVM_MEMFILL:
  case CL_COMMAND_MEMFILL_INTEL:
    svmFill(cmd->svm_fill.svm_ptr, cmd->svm_fill.size, cmd->svm_fill.pattern,
            cmd->svm_fill.pattern_size);
    *Msg = "Event SVM MemFill           ";
    break;

  case CL_COMMAND_SVM_MIGRATE_MEM:
  case CL_COMMAND_MIGRATEMEM_INTEL:
    svmMigrate(cmd->svm_migrate.num_svm_pointers,
               cmd->svm_migrate.svm_pointers,
               cmd->svm_migrate.sizes);
    *Msg = "Event SVM Migrate_Mem       ";
    break;

  case CL_COMMAND_MEMADVISE_INTEL:
    svmAdvise(cmd->mem_advise.ptr, cmd->mem_advise.size,
              cmd->mem_advise.advice);
    *Msg = "Event SVM Mem_Advise        ";
    break;

  case CL_COMMAND_COMMAND_BUFFER_KHR:
    *Msg = "Command Buffer KHR          ";
    break;

  default:
    POCL_ABORT_UNIMPLEMENTED("An unknown command type");
    break;
  }
}

void Level0Queue::allocNextFreeEvent() {
  if (!isCommandListActive)
    return;
  PreviousEventH = CurrentEventH;

  assert(!AvailableDeviceEvents.empty());
  CurrentEventH = AvailableDeviceEvents.front();
  AvailableDeviceEvents.pop();
  DeviceEventsToReset.push(CurrentEventH);
}

void Level0Queue::execCommand(_cl_command_node *Cmd) {

  cl_event event = Cmd->sync.event.event;
  ze_result_t res;

  const char *Msg = nullptr;
  pocl_update_event_running(event);

#ifndef LEVEL0_IMMEDIATE_CMDLIST
  LEVEL0_CHECK_ABORT(zeCommandListReset(CmdListH));
#endif
  uint64_t HostStartTS = pocl_gettimemono_ns();

  CurrentEventH = nullptr;
  PreviousEventH = nullptr;
  isCommandListActive = false;
  appendEventToList(Cmd, &Msg);

#ifndef LEVEL0_IMMEDIATE_CMDLIST
  LEVEL0_CHECK_ABORT(zeCommandListAppendBarrier(CmdListH, nullptr, 0, nullptr));
  LEVEL0_CHECK_ABORT(zeCommandListClose(CmdListH));
  LEVEL0_CHECK_ABORT(
      zeCommandQueueExecuteCommandLists(QueueH, 1, &CmdListH, nullptr));
  LEVEL0_CHECK_ABORT(
      zeCommandQueueSynchronize(QueueH, std::numeric_limits<uint64_t>::max()));
#endif

  uint64_t HostFinishTS = pocl_gettimemono_ns();
  event->time_start = HostStartTS;
  event->time_end = HostFinishTS;

  POCL_UPDATE_EVENT_COMPLETE_MSG(event, Msg);
}

void Level0Queue::execCommandBatch(BatchType &Batch) {

  ze_result_t res;

  LEVEL0_CHECK_ABORT(zeCommandListReset(CmdListH));

  POCL_MEASURE_START(ZeListPrepare);

  CurrentEventH = nullptr;
  PreviousEventH = nullptr;
  isCommandListActive = true;
  const char *Msg = nullptr;
  std::deque<const char *> Msgs;
  for (auto E : Batch) {
    _cl_command_node *Cmd = E->command;
    appendEventToList(Cmd, &Msg);
    Msgs.push_back(Msg);
  }

  res = zeCommandListAppendBarrier(CmdListH,
                                   nullptr, // signal event
                                   CurrentEventH ? 1 : 0,
                                   CurrentEventH ? &CurrentEventH : nullptr);
  LEVEL0_CHECK_ABORT(res);

  while (!DeviceEventsToReset.empty()) {
    ze_event_handle_t E = DeviceEventsToReset.front();
    DeviceEventsToReset.pop();
    LEVEL0_CHECK_ABORT(zeCommandListAppendEventReset(CmdListH, E));
    AvailableDeviceEvents.push(E);
  }
  LEVEL0_CHECK_ABORT(zeCommandListClose(CmdListH));

  POCL_MEASURE_FINISH(ZeListPrepare);
  POCL_MEASURE_START(ZeListExec);
  LEVEL0_CHECK_ABORT(
      zeCommandQueueExecuteCommandLists(QueueH, 1, &CmdListH, nullptr));
  for (auto E : Batch) {
    POCL_LOCK_OBJ(E);
    pocl_update_event_submitted(E);
    pocl_update_event_running_unlocked(E);
    POCL_UNLOCK_OBJ(E);
  }

  LEVEL0_CHECK_ABORT(
      zeCommandQueueSynchronize(QueueH, std::numeric_limits<uint64_t>::max()));

  POCL_MEASURE_FINISH(ZeListExec);
  for (auto E : Batch) {
    assert(!Msgs.empty());
    const char *Msg = Msgs.front();
    POCL_UPDATE_EVENT_COMPLETE_MSG(E, Msg);
    Msgs.pop_front();
  }
}

void Level0Queue::syncUseMemHostPtr(pocl_mem_identifier *MemId, cl_mem Mem,
                                    size_t Offset, size_t Size) {
  assert(Mem);

  if ((Mem->flags & CL_MEM_USE_HOST_PTR) == 0) {
    return;
  }

  char *DevPtr = static_cast<char *>(MemId->mem_ptr);
  char *MemHostPtr = static_cast<char *>(Mem->mem_host_ptr);

  // host visible mem = skip
  if (MemHostPtr == DevPtr) {
    return;
  }

  allocNextFreeEvent();
  LEVEL0_CHECK_ABORT(zeCommandListAppendMemoryCopy(
      CmdListH, MemHostPtr + Offset, DevPtr + Offset, Size, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr));
}

void Level0Queue::syncUseMemHostPtr(pocl_mem_identifier *MemId, cl_mem Mem,
                                    const size_t Origin[3],
                                    const size_t Region[3],
                                    size_t RowPitch,
                                    size_t SlicePitch) {
  assert(Mem);

  if ((Mem->flags & CL_MEM_USE_HOST_PTR) == 0) {
    return;
  }

  char *DevPtr = static_cast<char *>(MemId->mem_ptr);
  char *MemHostPtr = static_cast<char *>(Mem->mem_host_ptr);

  // host visible mem = skip
  if (DevPtr == MemHostPtr) {
    return;
  }

  ze_copy_region_t ZeRegion;
  ZeRegion.originX = Origin[0];
  ZeRegion.originY = Origin[1];
  ZeRegion.originZ = Origin[2];
  ZeRegion.width = Region[0];
  ZeRegion.height = Region[1];
  ZeRegion.depth = Region[2];

  ze_result_t res = zeCommandListAppendMemoryCopyRegion(
      CmdListH,
      MemHostPtr, &ZeRegion, RowPitch, SlicePitch,
      DevPtr, &ZeRegion, RowPitch, SlicePitch,
      nullptr, 0, nullptr);
  LEVEL0_CHECK_ABORT(res);
}

void Level0Queue::read(void *__restrict__ HostPtr,
                       pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
                       size_t Offset, size_t Size) {
  char *DevPtr = static_cast<char *>(SrcMemId->mem_ptr);
  if ((DevPtr + Offset) == HostPtr) {
    // this can happen when coming from CL_COMMAND_MIGRATE_MEM_OBJECTS
    POCL_MSG_WARN("Read skipped, HostPtr == DevPtr\n");
    return;
  }
  POCL_MSG_PRINT_LEVEL0("READ from %p OFF %zu SIZE %zu \n",
                        HostPtr, Offset, Size);
  allocNextFreeEvent();
  LEVEL0_CHECK_ABORT(zeCommandListAppendMemoryCopy(
      CmdListH, HostPtr, DevPtr + Offset, Size, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr));
}

void Level0Queue::write(const void *__restrict__ HostPtr,
                        pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                        size_t Offset, size_t Size) {
  char *DevPtr = static_cast<char *>(DstMemId->mem_ptr);
  if ((DevPtr + Offset) == HostPtr) {
    // this can happen when coming from CL_COMMAND_MIGRATE_MEM_OBJECTS
    POCL_MSG_WARN("Write skipped, HostPtr == DevPtr\n");
    return;
  }

  POCL_MSG_PRINT_LEVEL0("WRITE to %p OFF %zu SIZE %zu\n",
                        HostPtr, Offset, Size);
  allocNextFreeEvent();
  LEVEL0_CHECK_ABORT(zeCommandListAppendMemoryCopy(CmdListH, DevPtr + Offset,
         HostPtr, Size, CurrentEventH, PreviousEventH ? 1 : 0,
         PreviousEventH ? &PreviousEventH : nullptr));
}

void Level0Queue::copy(pocl_mem_identifier *DstMemDd, cl_mem DstBuf,
                       pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
                       size_t DstOffset, size_t SrcOffset, size_t Size) {
  char *SrcPtr = static_cast<char *>(SrcMemId->mem_ptr);
  char *DstPtr = static_cast<char *>(DstMemDd->mem_ptr);
  POCL_MSG_PRINT_LEVEL0("COPY | SRC %p OFF %zu | DST %p OFF %zu | SIZE %zu\n",
                        SrcPtr, SrcOffset, DstPtr, DstOffset, Size);
  allocNextFreeEvent();
  LEVEL0_CHECK_ABORT(zeCommandListAppendMemoryCopy(
      CmdListH, DstPtr + DstOffset, SrcPtr + SrcOffset, Size, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr));
}

void Level0Queue::copyRect(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                           pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
                           const size_t *__restrict__ const DstOrigin,
                           const size_t *__restrict__ const SrcOrigin,
                           const size_t *__restrict__ const Region,
                           size_t const DstRowPitch, size_t const DstSlicePitch,
                           size_t const SrcRowPitch,
                           size_t const SrcSlicePitch) {
  char *SrcPtr = static_cast<char *>(SrcMemId->mem_ptr);
  char *DstPtr = static_cast<char *>(DstMemId->mem_ptr);
  POCL_MSG_PRINT_LEVEL0("COPY RECT | SRC %p | DST %p \n", SrcPtr, DstPtr);

  ze_copy_region_t DstRegion;
  ze_copy_region_t SrcRegion;
  SrcRegion.originX = SrcOrigin[0];
  SrcRegion.originY = SrcOrigin[1];
  SrcRegion.originZ = SrcOrigin[2];
  SrcRegion.width = Region[0];
  SrcRegion.height = Region[1];
  SrcRegion.depth = Region[2];
  DstRegion.originX = DstOrigin[0];
  DstRegion.originY = DstOrigin[1];
  DstRegion.originZ = DstOrigin[2];
  DstRegion.width = Region[0];
  DstRegion.height = Region[1];
  DstRegion.depth = Region[2];

  allocNextFreeEvent();
  ze_result_t res = zeCommandListAppendMemoryCopyRegion(
      CmdListH, DstPtr, &DstRegion, DstRowPitch, DstSlicePitch, SrcPtr,
      &SrcRegion, SrcRowPitch, SrcSlicePitch, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);
  LEVEL0_CHECK_ABORT(res);
}

void Level0Queue::readRect(void *__restrict__ HostPtr,
                           pocl_mem_identifier *SrcMemId, cl_mem SrcBuf,
                           const size_t *__restrict__ const BufferOrigin,
                           const size_t *__restrict__ const HostOrigin,
                           const size_t *__restrict__ const Region,
                           size_t const BufferRowPitch,
                           size_t const BufferSlicePitch,
                           size_t const HostRowPitch,
                           size_t const HostSlicePitch) {
  const char *BufferPtr = static_cast<const char *>(SrcMemId->mem_ptr);

  ze_copy_region_t HostRegion;
  ze_copy_region_t BufferRegion;
  BufferRegion.originX = BufferOrigin[0];
  BufferRegion.originY = BufferOrigin[1];
  BufferRegion.originZ = BufferOrigin[2];
  BufferRegion.width = Region[0];
  BufferRegion.height = Region[1];
  BufferRegion.depth = Region[2];
  HostRegion.originX = HostOrigin[0];
  HostRegion.originY = HostOrigin[1];
  HostRegion.originZ = HostOrigin[2];
  HostRegion.width = Region[0];
  HostRegion.height = Region[1];
  HostRegion.depth = Region[2];

  allocNextFreeEvent();
  ze_result_t res = zeCommandListAppendMemoryCopyRegion(
      CmdListH, HostPtr, &HostRegion, HostRowPitch, HostSlicePitch, BufferPtr,
      &BufferRegion, BufferRowPitch, BufferSlicePitch, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);
  LEVEL0_CHECK_ABORT(res);
}

void Level0Queue::writeRect(const void *__restrict__ HostPtr,
                            pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                            const size_t *__restrict__ const BufferOrigin,
                            const size_t *__restrict__ const HostOrigin,
                            const size_t *__restrict__ const Region,
                            size_t const BufferRowPitch,
                            size_t const BufferSlicePitch,
                            size_t const HostRowPitch,
                            size_t const HostSlicePitch) {
  char *BufferPtr = static_cast<char *>(DstMemId->mem_ptr);

  ze_copy_region_t HostRegion;
  ze_copy_region_t BufferRegion;
  BufferRegion.originX = BufferOrigin[0];
  BufferRegion.originY = BufferOrigin[1];
  BufferRegion.originZ = BufferOrigin[2];
  BufferRegion.width = Region[0];
  BufferRegion.height = Region[1];
  BufferRegion.depth = Region[2];
  HostRegion.originX = HostOrigin[0];
  HostRegion.originY = HostOrigin[1];
  HostRegion.originZ = HostOrigin[2];
  HostRegion.width = Region[0];
  HostRegion.height = Region[1];
  HostRegion.depth = Region[2];

  allocNextFreeEvent();
  ze_result_t res = zeCommandListAppendMemoryCopyRegion(
      CmdListH, BufferPtr, &BufferRegion, BufferRowPitch, BufferSlicePitch,
      HostPtr, &HostRegion, HostRowPitch, HostSlicePitch, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);
  LEVEL0_CHECK_ABORT(res);
}

void Level0Queue::memfillImpl(Level0Device *Device,
                              ze_command_list_handle_t CmdListH,
                              const void *MemPtr, size_t Size, size_t Offset,
                              const void *__restrict__ Pattern,
                              size_t PatternSize) {

  ze_kernel_handle_t KernelH = nullptr;
  ze_module_handle_t ModuleH = nullptr;
  Level0Kernel *Ker = nullptr;
  bool Res = Device->getMemfillKernel(PatternSize, &Ker, ModuleH, KernelH);
  assert(Res == true);
  assert(KernelH);
  assert(ModuleH);

  // TODO this might be not enough: we might need to hold the lock until after
  // zeQueueSubmit
  std::lock_guard<std::mutex> KernelLockGuard(Ker->getMutex());

  // set kernel arg 0 = mem pointer
  ze_result_t ZeRes =
      zeKernelSetArgumentValue(KernelH, 0, sizeof(void *), &MemPtr);
  LEVEL0_CHECK_ABORT(ZeRes);

  // set kernel arg 1 = pattern (POD type)
  ZeRes = zeKernelSetArgumentValue(KernelH, 1, PatternSize, Pattern);
  LEVEL0_CHECK_ABORT(ZeRes);

  uint32_t TotalWGsX = Size / PatternSize;
  uint32_t OffsetX = Offset / PatternSize;
  uint32_t WGSizeX = 1;

  // TODO fix to have higher utilization
  uint32_t MaxWG = Device->getMaxWGSize() / 2;
  while ((TotalWGsX > 1) && ((TotalWGsX & 1) == 0) && (WGSizeX <= MaxWG)) {
    TotalWGsX /= 2;
    WGSizeX *= 2;
  }

  if (Offset) {
    if (Device->supportsGlobalOffsets()) {
      ZeRes = zeKernelSetGlobalOffsetExp(KernelH, OffsetX, 0, 0);
      LEVEL0_CHECK_ABORT(ZeRes);
    } else {
      POCL_MSG_ERR("memfill: offset specified but device doesn't "
                   "support Global offsets\n");
    }
  }

  ZeRes = zeKernelSetGroupSize(KernelH, WGSizeX, 1, 1);
  LEVEL0_CHECK_ABORT(ZeRes);
  ze_group_count_t LaunchFuncArgs = {TotalWGsX, 1, 1};
  allocNextFreeEvent();
  ZeRes = zeCommandListAppendLaunchKernel(
      CmdListH, KernelH, &LaunchFuncArgs, CurrentEventH, PreviousEventH ? 1 : 0,
      PreviousEventH ? &PreviousEventH : nullptr);

  LEVEL0_CHECK_ABORT(ZeRes);
}

void Level0Queue::memFill(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                          size_t Size, size_t Offset,
                          const void *__restrict__ Pattern,
                          size_t PatternSize) {
  char *DstPtr = static_cast<char *>(DstMemId->mem_ptr);
  POCL_MSG_PRINT_LEVEL0("MEMFILL | PTR %p | SIZE %zu | PAT SIZE %zu\n", DstPtr,
                        Size, PatternSize);
  memfillImpl(Device, CmdListH, DstPtr, Size, Offset, Pattern, PatternSize);
}


void Level0Queue::mapMem(pocl_mem_identifier *SrcMemId,
                         cl_mem SrcBuf, mem_mapping_t *Map) {
  char *SrcPtr = static_cast<char *>(SrcMemId->mem_ptr);

  POCL_MSG_PRINT_LEVEL0("MAP MEM: %p FLAGS %zu\n", SrcPtr, Map->map_flags);

  if ((Map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION) != 0u) {
    return;
  }

  assert(SrcBuf);
  // host visible mem == skip
  if (SrcBuf->mem_host_ptr == SrcMemId->mem_ptr) {
    assert(Map->host_ptr == (SrcPtr + Map->offset));
    return;
  }

  // memcpy (map->HostPtr, src_device_ptr + map->offset, map->size);
  allocNextFreeEvent();
  ze_result_t res = zeCommandListAppendMemoryCopy(
      CmdListH, Map->host_ptr, SrcPtr + Map->offset, Map->size, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);
  LEVEL0_CHECK_ABORT(res);
}

void Level0Queue::unmapMem(pocl_mem_identifier *DstMemId, cl_mem DstBuf,
                           mem_mapping_t *Map) {
  char *DstPtr = static_cast<char *>(DstMemId->mem_ptr);

  POCL_MSG_PRINT_LEVEL0("UNMAP MEM: %p FLAGS %zu\n", DstPtr, Map->map_flags);

  // for read mappings, don't copy anything
  if (Map->map_flags == CL_MAP_READ) {
    return;
  }

  assert(DstBuf);
  // host visible mem == skip
  if (DstBuf->mem_host_ptr == DstMemId->mem_ptr) {
    assert(Map->host_ptr == (DstPtr + Map->offset));
    return;
  }

  allocNextFreeEvent();
  // memcpy (dst_device_ptr + map->offset, map->HostPtr, map->size);
  ze_result_t res = zeCommandListAppendMemoryCopy(
      CmdListH, DstPtr + Map->offset, Map->host_ptr, Map->size, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);
  LEVEL0_CHECK_ABORT(res);
}

void Level0Queue::copyImageRect(cl_mem SrcImage, cl_mem DstImage,
                                pocl_mem_identifier *SrcMemId,
                                pocl_mem_identifier *DstMemId,
                                const size_t *SrcOrigin,
                                const size_t *DstOrigin, const size_t *Region) {

  ze_image_handle_t SrcImg =
      static_cast<ze_image_handle_t>(SrcMemId->extra_ptr);
  ze_image_handle_t DstImg =
      static_cast<ze_image_handle_t>(DstMemId->extra_ptr);
  POCL_MSG_PRINT_LEVEL0("COPY IMAGE RECT | SRC %p | DST %p \n", (void *)SrcImg,
                        (void *)DstImg);

  ze_image_region_t DstRegion;
  ze_image_region_t SrcRegion;
  SrcRegion.originX = SrcOrigin[0];
  SrcRegion.originY = SrcOrigin[1];
  SrcRegion.originZ = SrcOrigin[2];
  SrcRegion.width = Region[0];
  SrcRegion.height = Region[1];
  SrcRegion.depth = Region[2];
  DstRegion.originX = DstOrigin[0];
  DstRegion.originY = DstOrigin[1];
  DstRegion.originZ = DstOrigin[2];
  DstRegion.width = Region[0];
  DstRegion.height = Region[1];
  DstRegion.depth = Region[2];

  allocNextFreeEvent();
  ze_result_t Res = zeCommandListAppendImageCopyRegion(
      CmdListH, DstImg, SrcImg, &DstRegion, &SrcRegion, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);

  LEVEL0_CHECK_ABORT(Res);
}

void Level0Queue::writeImageRect(cl_mem DstImage, pocl_mem_identifier *DstMemId,
                                 const void *__restrict__ SrcHostPtr,
                                 pocl_mem_identifier *SrcMemId,
                                 const size_t *Origin, const size_t *Region,
                                 size_t SrcRowPitch, size_t SrcSlicePitch,
                                 size_t SrcOffset) {
  const char *SrcPtr = nullptr;
  if (SrcHostPtr != nullptr) {
    SrcPtr = static_cast<const char *>(SrcHostPtr) + SrcOffset;
  } else {
    assert(SrcMemId);
    SrcPtr = static_cast<const char *>(SrcMemId->mem_ptr) + SrcOffset;
  }

  ze_image_handle_t DstImg =
      static_cast<ze_image_handle_t>(DstMemId->extra_ptr);
  POCL_MSG_PRINT_LEVEL0("COPY IMAGE RECT | SRC PTR %p | DST IMG %p \n",
                        (void *)SrcPtr, (void *)DstImg);

  ze_image_region_t DstRegion;
  DstRegion.originX = Origin[0];
  DstRegion.originY = Origin[1];
  DstRegion.originZ = Origin[2];
  DstRegion.width = Region[0];
  DstRegion.height = Region[1];
  DstRegion.depth = Region[2];

  // unfortunately, this returns ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
  //  ze_result_t Res = zeCommandListAppendImageCopyFromMemoryExt(CmdListH,
  //  DstImg, SrcPtr, &DstRegion,
  //                                            SrcRowPitch, SrcSlicePitch,
  //                                            nullptr, 0, nullptr);
  allocNextFreeEvent();
  ze_result_t Res = zeCommandListAppendImageCopyFromMemory(
      CmdListH, DstImg, SrcPtr, &DstRegion, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);
  LEVEL0_CHECK_ABORT(Res);
}

void Level0Queue::readImageRect(cl_mem SrcImage, pocl_mem_identifier *SrcMemId,
                                void *__restrict__ DstHostPtr,
                                pocl_mem_identifier *DstMemId,
                                const size_t *Origin, const size_t *Region,
                                size_t DstRowPitch, size_t DstSlicePitch,
                                size_t DstOffset) {
  char *DstPtr = nullptr;
  if (DstHostPtr != nullptr) {
    DstPtr = static_cast<char *>(DstHostPtr) + DstOffset;
  } else {
    assert(DstMemId);
    DstPtr = static_cast<char *>(DstMemId->mem_ptr) + DstOffset;
  }

  ze_image_handle_t SrcImg =
      static_cast<ze_image_handle_t>(SrcMemId->extra_ptr);
  POCL_MSG_PRINT_LEVEL0("COPY IMAGE RECT | SRC IMG %p | DST PTR %p \n",
                        (void *)SrcImg, (void *)DstPtr);

  ze_image_region_t SrcRegion;
  SrcRegion.originX = Origin[0];
  SrcRegion.originY = Origin[1];
  SrcRegion.originZ = Origin[2];
  SrcRegion.width = Region[0];
  SrcRegion.height = Region[1];
  SrcRegion.depth = Region[2];

  // unfortunately, this returns ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
  //  ze_result_t Res = zeCommandListAppendImageCopyToMemoryExt(CmdListH,
  //  DstPtr, SrcImg, &SrcRegion,
  //                                          DstRowPitch, DstSlicePitch,
  //                                          nullptr, 0, nullptr);
  allocNextFreeEvent();
  ze_result_t Res = zeCommandListAppendImageCopyToMemory(
      CmdListH, DstPtr, SrcImg, &SrcRegion, CurrentEventH,
      PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);

  LEVEL0_CHECK_ABORT(Res);
}

void Level0Queue::mapImage(pocl_mem_identifier *MemId,
                           cl_mem SrcImage, mem_mapping_t *Map) {

  char *SrcImgPtr = static_cast<char *>(MemId->mem_ptr);
  POCL_MSG_PRINT_LEVEL0("MAP IMAGE: %p FLAGS %zu\n", SrcImgPtr, Map->map_flags);

  if ((Map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION) != 0u) {
    return;
  }

  assert(Map->host_ptr == (SrcImgPtr + Map->offset));

  readImageRect(SrcImage, MemId, SrcImgPtr, nullptr, Map->origin, Map->region,
                Map->row_pitch, Map->slice_pitch, Map->offset);
}

void Level0Queue::unmapImage(pocl_mem_identifier *MemId,
                             cl_mem DstImage, mem_mapping_t *Map) {
  char *DstImgPtr = static_cast<char *>(MemId->mem_ptr);

  POCL_MSG_PRINT_LEVEL0("UNMAP IMAGE: %p FLAGS %zu\n", DstImgPtr,
                        Map->map_flags);

  // for read mappings, don't copy anything
  if (Map->map_flags == CL_MAP_READ) {
    return;
  }

  assert(Map->host_ptr == (DstImgPtr + Map->offset));

  writeImageRect(DstImage, MemId, DstImgPtr, nullptr, Map->origin, Map->region,
                 Map->row_pitch, Map->slice_pitch, Map->offset);
}

void Level0Queue::fillImage(cl_mem Image, pocl_mem_identifier *MemId,
                            const size_t *Origin, const size_t *Region,
                            cl_uint4 OrigPixel, pixel_t FillPixel,
                            size_t PixelSize) {
  char *MapPtr = static_cast<char *>(MemId->mem_ptr);
  ze_image_handle_t ImageH = (ze_image_handle_t)(MemId->extra_ptr);
  assert(Image);
  //" SIZE %zu | PAT SIZE %zu\n",
  POCL_MSG_PRINT_LEVEL0("IMAGEFILL | PTR %p | IMAGE %p |\n", MapPtr, ImageH);

  ze_kernel_handle_t KernelH = nullptr;
  ze_module_handle_t ModuleH = nullptr;
  Level0Kernel *Ker = nullptr;
  bool Res = Device->getImagefillKernel(Image->image_channel_order,
                                        Image->image_channel_data_type,
                                        Image->type, &Ker, ModuleH, KernelH);
  assert(Res == true);
  assert(KernelH);
  assert(ModuleH);

  // TODO this might be not enough: we might need to hold the lock until after
  // zeQueueSubmit
  std::lock_guard<std::mutex> KernelLockGuard(Ker->getMutex());

  // set kernel arg 0 = image pointer
  ze_result_t ZeRes =
      zeKernelSetArgumentValue(KernelH, 0, sizeof(ze_image_handle_t), &ImageH);
  LEVEL0_CHECK_ABORT(ZeRes);

  // set kernel arg 1 = Pixel pattern (POD type)
  ZeRes = zeKernelSetArgumentValue(KernelH, 1, sizeof(pixel_t), FillPixel);
  LEVEL0_CHECK_ABORT(ZeRes);

  if (Origin[0] || Origin[1] || Origin[2]) {
    if (Device->supportsGlobalOffsets()) {
      ZeRes =
          zeKernelSetGlobalOffsetExp(KernelH, (uint32_t)Origin[0],
                                     (uint32_t)Origin[1], (uint32_t)Origin[2]);
      LEVEL0_CHECK_ABORT(ZeRes);
    } else {
      POCL_MSG_ERR("imagefill: origin specified but device doesn't "
                   "support Global offsets\n");
    }
  }

  // TODO could be better
  ZeRes = zeKernelSetGroupSize(KernelH, 1, 1, 1);
  LEVEL0_CHECK_ABORT(ZeRes);
  ze_group_count_t LaunchFuncArgs = {(uint32_t)Region[0], (uint32_t)Region[1],
                                     (uint32_t)Region[2]};
  allocNextFreeEvent();
  ZeRes = zeCommandListAppendLaunchKernel(
      CmdListH, KernelH, &LaunchFuncArgs, CurrentEventH, PreviousEventH ? 1 : 0,
      PreviousEventH ? &PreviousEventH : nullptr);

  LEVEL0_CHECK_ABORT(ZeRes);
}

void Level0Queue::svmMap(void *Ptr) {}

void Level0Queue::svmUnmap(void *Ptr) {}

void Level0Queue::svmCopy(void *DstPtr, const void *SrcPtr, size_t Size) {
  POCL_MSG_PRINT_LEVEL0("SVM COPY | SRC %p | DST %p | SIZE %zu\n", SrcPtr,
                        DstPtr, Size);

  allocNextFreeEvent();
  ze_result_t res = zeCommandListAppendMemoryCopy(
      CmdListH, DstPtr, SrcPtr, Size, CurrentEventH, PreviousEventH ? 1 : 0,
      PreviousEventH ? &PreviousEventH : nullptr);
  LEVEL0_CHECK_ABORT(res);
}

void Level0Queue::svmFill(void *DstPtr, size_t Size, void *Pattern,
                          size_t PatternSize) {
  POCL_MSG_PRINT_LEVEL0("SVM FILL | PTR %p | SIZE %zu | PAT SIZE %zu\n", DstPtr,
                        Size, PatternSize);

  memfillImpl(Device, CmdListH, DstPtr, Size, 0, Pattern, PatternSize);

#if 0
  // this *might* be useful some way (perhaps faster), but:
  // 1) some devices (Arc A750) have insufficient limit on pattern size (16)
  // 2) it seems to have a bug that causes a failure with pattern size 2
  //    ... on test Unit_hipMemset_SetMemoryWithOffset

  ze_result_t Res = zeCommandListAppendMemoryFill(
      CmdListH, DstPtr, Pattern, PatternSize, Size, nullptr, 0, nullptr);
  LEVEL0_CHECK_ABORT(Res);
#endif
}

// The function clEnqueueMigrateMemINTEL explicitly migrates a region of
// a shared Unified Shared Memory allocation to the device associated
// with command_queue. This is a hint that may improve performance and
// is not required for correctness
void Level0Queue::svmMigrate(unsigned num_svm_pointers, void **svm_pointers,
                             size_t *sizes) {
  for (unsigned i = 0; i < num_svm_pointers; ++i) {
    ze_result_t Res =
        zeCommandListAppendMemoryPrefetch(CmdListH, svm_pointers[i], sizes[i]);
    LEVEL0_CHECK_ABORT(Res);
  }
}

void Level0Queue::svmAdvise(const void *ptr, size_t size,
                            cl_mem_advice_intel advice) {
  // TODO convert cl_advice to ZeAdvice. The current API doesn't
  // seem to specify any valid values
  if (advice == 0)
    return;
  else
    POCL_MSG_ERR("svmAdvise: unknown advice value %zu\n", (size_t)advice);
  ze_memory_advice_t ZeAdvice = ZE_MEMORY_ADVICE_BIAS_UNCACHED;
  ze_result_t Res = zeCommandListAppendMemAdvise(
      CmdListH, Device->getDeviceHandle(), ptr, size, ZeAdvice);
  LEVEL0_CHECK_ABORT(Res);
}

bool Level0Queue::setupKernelArgs(ze_module_handle_t ModuleH,
                                  ze_kernel_handle_t KernelH, cl_device_id Dev,
                                  unsigned DeviceI, _cl_command_run *RunCmd) {
  cl_kernel Kernel = RunCmd->kernel;
  struct pocl_argument *PoclArg = RunCmd->arguments;

  // static locals are taken care of in ZE compiler
  assert(Kernel->meta->num_locals == 0);

  cl_uint i = 0;
  ze_result_t Res = ZE_RESULT_SUCCESS;
  for (i = 0; i < Kernel->meta->num_args; ++i) {
    if (ARG_IS_LOCAL(Kernel->meta->arg_info[i])) {
      assert(PoclArg[i].size > 0);
      Res = zeKernelSetArgumentValue(KernelH, i, PoclArg[i].size, NULL);
      LEVEL0_CHECK_ABORT(Res);

    } else if (Kernel->meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      assert(PoclArg[i].size == sizeof(void *));

      if (PoclArg[i].value == NULL) {
        Res = zeKernelSetArgumentValue(KernelH, i, sizeof(void *), nullptr);
      } else if (PoclArg[i].is_raw_ptr != 0) {
        void *MemPtr = *(void**)PoclArg[i].value;
        if (MemPtr == nullptr)
          Res = zeKernelSetArgumentValue(KernelH, i, sizeof(void *), nullptr);
        else
          Res = zeKernelSetArgumentValue(KernelH, i, sizeof(void *), &MemPtr);
      } else {
        cl_mem arg_buf = (*(cl_mem *)(PoclArg[i].value));
        pocl_mem_identifier *memid = &arg_buf->device_ptrs[Dev->global_mem_id];
        void *MemPtr = memid->mem_ptr;
        Res = zeKernelSetArgumentValue(KernelH, i, sizeof(void *), &MemPtr);
        LEVEL0_CHECK_ABORT(Res);
        // optimization for read-only buffers
        ze_memory_advice_t Adv =
            (PoclArg[i].is_readonly ? ZE_MEMORY_ADVICE_SET_READ_MOSTLY
                                    : ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY);
        Res = zeCommandListAppendMemAdvise(CmdListH, Device->getDeviceHandle(),
                                           MemPtr, arg_buf->size, Adv);
      }
      LEVEL0_CHECK_ABORT(Res);

    } else if (Kernel->meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      assert(PoclArg[i].value != NULL);
      assert(PoclArg[i].size == sizeof(void *));

      cl_mem arg_buf = (*(cl_mem *)(PoclArg[i].value));
      pocl_mem_identifier *memid = &arg_buf->device_ptrs[Dev->global_mem_id];
      void *hImage = memid->extra_ptr;
      Res = zeKernelSetArgumentValue(KernelH, i, sizeof(void *), &hImage);
      LEVEL0_CHECK_ABORT(Res);
    } else if (Kernel->meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      assert(PoclArg[i].value != NULL);
      assert(PoclArg[i].size == sizeof(void *));

      cl_sampler sam = (*(cl_sampler *)(PoclArg[i].value));
      ze_sampler_handle_t hSampler =
          (ze_sampler_handle_t)sam->device_data[Dev->dev_id];

      Res = zeKernelSetArgumentValue(KernelH, i, sizeof(void *), &hSampler);
      LEVEL0_CHECK_ABORT(Res);
    } else {
      assert(PoclArg[i].value != NULL);
      assert(PoclArg[i].size > 0);
      assert(PoclArg[i].size == Kernel->meta->arg_info[i].type_size);

      Res = zeKernelSetArgumentValue(KernelH, i, PoclArg[i].size,
                                     PoclArg[i].value);
      LEVEL0_CHECK_ABORT(Res);
    }
  }
  return false;
}

void Level0Queue::runWithOffsets(struct pocl_context *PoclCtx,
                                 ze_kernel_handle_t KernelH) {
  ze_result_t Res = ZE_RESULT_SUCCESS;
  uint32_t StartOffsetX = PoclCtx->global_offset[0];
  uint32_t StartOffsetY = PoclCtx->global_offset[1];
  uint32_t StartOffsetZ = PoclCtx->global_offset[2];

  uint32_t WGSizeX = PoclCtx->local_size[0];
  uint32_t WGSizeY = PoclCtx->local_size[1];
  uint32_t WGSizeZ = PoclCtx->local_size[2];

  uint32_t TotalWGsX = PoclCtx->num_groups[0];
  uint32_t TotalWGsY = PoclCtx->num_groups[1];
  uint32_t TotalWGsZ = PoclCtx->num_groups[2];

  uint32_t CurrentWGsX = 0;
  uint32_t CurrentWGsY = 0;
  uint32_t CurrentWGsZ = 0;
  uint32_t CurrentOffsetX = 0;
  uint32_t CurrentOffsetY = 0;
  uint32_t CurrentOffsetZ = 0;

  for (uint32_t OffsetZ = 0; OffsetZ < TotalWGsZ;
       OffsetZ += DeviceMaxWGSizes.s[2]) {
    CurrentWGsZ = std::min(DeviceMaxWGSizes.s[2], TotalWGsZ - OffsetZ);
    CurrentOffsetZ = StartOffsetZ + OffsetZ * WGSizeZ;
    {
      for (uint32_t OffsetY = 0; OffsetY < TotalWGsY;
           OffsetY += DeviceMaxWGSizes.s[1]) {
        CurrentWGsY = std::min(DeviceMaxWGSizes.s[1], TotalWGsY - OffsetY);
        CurrentOffsetY = StartOffsetY + OffsetY * WGSizeY;
        {
          for (uint32_t OffsetX = 0; OffsetX < TotalWGsX;
               OffsetX += DeviceMaxWGSizes.s[0]) {
            CurrentWGsX = std::min(DeviceMaxWGSizes.s[0], TotalWGsX - OffsetX);
            CurrentOffsetX = StartOffsetX + OffsetX * WGSizeX;

#if 0
            // debug code
            POCL_MSG_PRINT_LEVEL0(
               "WGs X %u Y %u Z %u ||| OFFS X %u Y %u Z %u ||| LOCAL X %u Y "
               "%u Z %u\n", TotalWGsX, TotalWGsY, TotalWGsZ, CurrentOffsetX,
               CurrentOffsetY, CurrentOffsetZ, CurrentWGsX, CurrentWGsY,
               CurrentWGsZ);
#endif
            Res = zeKernelSetGlobalOffsetExp(KernelH, CurrentOffsetX,
                                             CurrentOffsetY, CurrentOffsetZ);
            LEVEL0_CHECK_ABORT(Res);

            ze_group_count_t LaunchFuncArgs = {CurrentWGsX, CurrentWGsY,
                                               CurrentWGsZ};

            // TODO this can actually be executed in parallel.
            allocNextFreeEvent();
            Res = zeCommandListAppendLaunchKernel(
                CmdListH, KernelH, &LaunchFuncArgs, CurrentEventH,
                PreviousEventH ? 1 : 0,
                PreviousEventH ? &PreviousEventH : nullptr);

            LEVEL0_CHECK_ABORT(Res);

            // TODO find out if there is a limit on number of
            // submitted commands in a single command list.
          }
        }
      }
    }
  }
}

void Level0Queue::run(_cl_command_node *Cmd) {
  cl_event Event = Cmd->sync.event.event;
  _cl_command_run *RunCmd = &Cmd->command.run;
  cl_device_id Dev = Cmd->device;
  assert(Cmd->type == CL_COMMAND_NDRANGE_KERNEL);
  cl_kernel Kernel = Cmd->command.run.kernel;
  cl_program Program = Kernel->program;
  unsigned DeviceI = Cmd->program_device_i;
  struct pocl_context *PoclCtx = &RunCmd->pc;

  assert(Program->data[DeviceI] != nullptr);
  Level0Program *L0Program = (Level0Program *)Program->data[DeviceI];
  assert(Kernel->data[DeviceI] != nullptr);
  Level0Kernel *L0Kernel = (Level0Kernel *)Kernel->data[DeviceI];

  bool Needs64bitPtrs = false;
  for (size_t i = 0; i < Event->num_buffers; ++i) {
    if (Event->mem_objs[i]->size > UINT32_MAX) {
      Needs64bitPtrs = true;
      break;
    }
  }

  unsigned TotalLocalWGSize =
      PoclCtx->local_size[0] * PoclCtx->local_size[1] * PoclCtx->local_size[2];
  ze_kernel_handle_t KernelH = nullptr;
  ze_module_handle_t ModuleH = nullptr;
  bool Res = Device->getBestKernel(L0Program, L0Kernel, Needs64bitPtrs,
                                   TotalLocalWGSize, ModuleH, KernelH);
  assert(Res == true);
  assert(KernelH);
  assert(ModuleH);

  // TODO this lock should be moved not re-locked
  // necessary to lock the kernel, since we're setting up kernel arguments
  // setting WG sizes and so on; this lock is released after
  // zeCommandListAppendKernel
  // TODO this might be not enough: we might need to hold the lock until after
  // zeQueueSubmit
  std::lock_guard<std::mutex> KernelLockGuard(L0Kernel->getMutex());

  ze_kernel_indirect_access_flags_t Flags = L0Kernel->getIndirectFlags();
  if (Flags != 0) {
    ze_result_t Res = zeKernelSetIndirectAccess(KernelH, Flags);
    LEVEL0_CHECK_ABORT(Res);
  }
  const std::map<void *, size_t> &AccessedPointers =
      L0Kernel->getAccessedPointers();
  for (auto &I : AccessedPointers) {
    void *Ptr = I.first;
    size_t Size = I.second;
    POCL_MSG_PRINT_MEMORY("Level0: Making %p (size %zu) resident.\n", Ptr,
                          Size);
    ze_result_t Res = zeContextMakeMemoryResident(
        Device->getContextHandle(), Device->getDeviceHandle(), Ptr, Size);
    LEVEL0_CHECK_ABORT(Res);
  }

  if (setupKernelArgs(ModuleH, KernelH, Dev, Cmd->program_device_i, RunCmd)) {
    POCL_MSG_ERR("Level0: Failed to setup kernel arguments\n");
    return;
  }

  uint32_t TotalWGsX = PoclCtx->num_groups[0];
  uint32_t TotalWGsY = PoclCtx->num_groups[1];
  uint32_t TotalWGsZ = PoclCtx->num_groups[2];
  size_t TotalWGs = TotalWGsX * TotalWGsY * TotalWGsZ;
  if (TotalWGs == 0) {
    return;
  }

  uint32_t WGSizeX = PoclCtx->local_size[0];
  uint32_t WGSizeY = PoclCtx->local_size[1];
  uint32_t WGSizeZ = PoclCtx->local_size[2];
  zeKernelSetGroupSize(KernelH, WGSizeX, WGSizeY, WGSizeZ);

  uint32_t StartOffsetX = PoclCtx->global_offset[0];
  uint32_t StartOffsetY = PoclCtx->global_offset[1];
  uint32_t StartOffsetZ = PoclCtx->global_offset[2];
  bool NeedsGlobalOffset = (StartOffsetX | StartOffsetY | StartOffsetZ) > 0;

  if (Device->supportsGlobalOffsets() && NeedsGlobalOffset) {
    runWithOffsets(PoclCtx, KernelH);
  } else {
    assert(!NeedsGlobalOffset &&
           "command needs "
           "global offsets, but device doesn't support them");
    ze_group_count_t LaunchFuncArgs = {TotalWGsX, TotalWGsY, TotalWGsZ};
    allocNextFreeEvent();
    ze_result_t ZeRes = zeCommandListAppendLaunchKernel(
        CmdListH, KernelH, &LaunchFuncArgs, CurrentEventH,
        PreviousEventH ? 1 : 0, PreviousEventH ? &PreviousEventH : nullptr);

    LEVEL0_CHECK_ABORT(ZeRes);
  }

  // zeKernelSetCacheConfig();
  // zeKernelSetIndirectAccess()
}

Level0Queue::Level0Queue(Level0WorkQueueInterface *WH,
                         ze_command_queue_handle_t Q,
                         ze_command_list_handle_t L, ze_event_pool_handle_t E,
                         uint32_t EvPoolSize, Level0Device *D) {

  WorkHandler = WH;
  QueueH = Q;
  CmdListH = L;
  Device = D;
  EvtPoolH = E;

  uint32_t TimeStampBits, KernelTimeStampBits;
  Device->getTimingInfo(TimeStampBits, KernelTimeStampBits, DeviceFrequency,
                        DeviceNsPerCycle);
  DeviceMaxValidTimestamp = (1UL << TimeStampBits) - 1;
  DeviceMaxValidKernelTimestamp = (1UL << KernelTimeStampBits) - 1;
  // since the value will be in NS, and unavoidably there will be some noise,
  // this slightly lowers the wrapping limit.
  uint64_t TimeStampWrapLimit = DeviceMaxValidTimestamp * 15 / 16;
  uint64_t KernelTimeStampWrapLimit = DeviceMaxValidKernelTimestamp * 15 / 16;
  // convert to nanoseconds
  DeviceTimerWrapTimeNs =
      (uint64_t)((double)TimeStampWrapLimit * DeviceNsPerCycle);
  DeviceKernelTimerWrapTimeNs =
      (uint64_t)((double)KernelTimeStampWrapLimit * DeviceNsPerCycle);

  Device->getMaxWGs(&DeviceMaxWGSizes);

  unsigned Idx = 0;
  unsigned NumDevEvents = EvPoolSize;
  for (Idx = 0; Idx < NumDevEvents; ++Idx) {

    ze_event_desc_t eventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr, // pNext
        Idx,     // index
        0,       // flags on signal
        ZE_EVENT_SCOPE_FLAG_SUBDEVICE |
            ZE_EVENT_SCOPE_FLAG_DEVICE // flags on wait
    };

    ze_event_handle_t EvH = nullptr;
    LEVEL0_CHECK_ABORT(zeEventCreate(EvtPoolH, &eventDesc, &EvH));
    AvailableDeviceEvents.push(EvH);
  }

  Thread = std::thread(&Level0Queue::runThread, this);
}

Level0Queue::~Level0Queue() {
  if (Thread.joinable()) {
    Thread.join();
  }
  assert(DeviceEventsToReset.empty());
  while (!AvailableDeviceEvents.empty()) {
    ze_event_handle_t EvH = AvailableDeviceEvents.front();
    AvailableDeviceEvents.pop();
    zeEventDestroy(EvH);
  }
  if (CmdListH != nullptr) {
    zeCommandListDestroy(CmdListH);
  }
  if (QueueH != nullptr) {
    zeCommandQueueDestroy(QueueH);
  }
  if (EvtPoolH != nullptr) {
    zeEventPoolDestroy(EvtPoolH);
  }
}

static constexpr unsigned CacheLineSize = 64;

bool Level0QueueGroup::init(unsigned Ordinal, unsigned Count,
                            Level0Device *Device) {

  ThreadExitRequested = false;

  ze_context_handle_t ContextH = Device->getContextHandle();
  ze_device_handle_t DeviceH = Device->getDeviceHandle();

  std::vector<ze_command_queue_handle_t> QHandles;
  std::vector<ze_command_list_handle_t> LHandles;
  std::vector<ze_event_pool_handle_t> EHandles;
  assert(Count > 0);
  QHandles.resize(Count);
  LHandles.resize(Count);
  EHandles.resize(Count);
  ze_result_t ZeRes = ZE_RESULT_SUCCESS;
  ze_command_queue_handle_t Queue = nullptr;
  ze_command_list_handle_t CmdList = nullptr;
  ze_event_pool_handle_t EventPool = nullptr;

  ze_event_pool_desc_t EvtPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
      0,            // flags
      EventPoolSize // num events
  };

#ifdef LEVEL0_IMMEDIATE_CMDLIST
  ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                          nullptr,
                                          Ordinal,
                                          0, // index
                                          0, // flags
                                          ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                          ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  for (unsigned i = 0; i < Count; ++i) {
    cmdQueueDesc.index = i;
    ZeRes = zeCommandListCreateImmediate(ContextH, DeviceH,
                               &cmdQueueDesc, &CmdList);
    LEVEL0_CHECK_RET(false, ZeRes);
    ZeRes = zeEventPoolCreate(ContextH, &EvtPoolDesc, 1, &DeviceH, &EventPool);
    LEVEL0_CHECK_RET(false, ZeRes);
    QHandles[i] = Queue;
    LHandles[i] = CmdList;
    EHandles[i] = EventPool;
  }
#else
  ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                          nullptr,
                                          Ordinal,
                                          0, // index
                                          0, // flags
                                          ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                          ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  ze_command_list_desc_t cmdListDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, Ordinal,
      ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING |
          ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT};

  for (unsigned i = 0; i < Count; ++i) {
    cmdQueueDesc.index = i;
    ZeRes = zeCommandQueueCreate(ContextH, DeviceH, &cmdQueueDesc, &Queue);
    LEVEL0_CHECK_RET(false, ZeRes);
    ZeRes = zeCommandListCreate(ContextH, DeviceH, &cmdListDesc, &CmdList);
    LEVEL0_CHECK_RET(false, ZeRes);
    ZeRes = zeEventPoolCreate(ContextH, &EvtPoolDesc, 1, &DeviceH, &EventPool);
    LEVEL0_CHECK_RET(false, ZeRes);
    QHandles[i] = Queue;
    LHandles[i] = CmdList;
    EHandles[i] = EventPool;
  }
#endif

  for (unsigned i = 0; i < Count; ++i) {
    Queues.emplace_back(new Level0Queue(
        this, QHandles[i], LHandles[i], EHandles[i], EventPoolSize, Device));
  }

  Available = true;
  return true;
}

Level0QueueGroup::~Level0QueueGroup() {
  std::unique_lock<std::mutex> Lock(Mutex);
  ThreadExitRequested = true;
  Cond.notify_all();
  Lock.unlock();
  Queues.clear();
}

void Level0QueueGroup::pushWork(_cl_command_node *Command) {
  std::lock_guard<std::mutex> Lock(Mutex);
  WorkQueue.push(Command);
  Cond.notify_one();
}

void Level0QueueGroup::pushCommandBatch(BatchType Batch) {
  std::lock_guard<std::mutex> Lock(Mutex);
  BatchWorkQueue.push(std::move(Batch));
  Cond.notify_one();
}

bool Level0QueueGroup::getWorkOrWait(_cl_command_node **Node,
                                     BatchType &Batch) {
  std::unique_lock<std::mutex> Lock(Mutex);
  *Node = nullptr;
  bool ShouldExit;
  do {

    ShouldExit = ThreadExitRequested;
    if (!WorkQueue.empty()) {
      *Node = WorkQueue.front();
      WorkQueue.pop();
      break;
    } else if (!BatchWorkQueue.empty()) {
      Batch = std::move(BatchWorkQueue.front());
      BatchWorkQueue.pop();
      break;
    } else {
      if (!ShouldExit) {
        Cond.wait(Lock);
      }
    }
  } while (!ShouldExit);

  Lock.unlock();
  return ShouldExit;
}

/// serialize SPIRV of the program since we might need
/// to rebuild it with new Spec Constants
/// also serialize the directory with native binaries
const char *LEVEL0_SERIALIZE_ENTRIES[3] = {"/program.bc", "/program.spv",
                                           "/native"};

static const cl_image_format SupportedImageFormats[] = {
    {CL_R, CL_SIGNED_INT8},        {CL_R, CL_SIGNED_INT16},
    {CL_R, CL_SIGNED_INT32},       {CL_R, CL_SNORM_INT8},
    {CL_R, CL_SNORM_INT16},        {CL_R, CL_UNSIGNED_INT8},
    {CL_R, CL_UNSIGNED_INT16},     {CL_R, CL_UNSIGNED_INT32},
    {CL_R, CL_UNORM_INT8},         {CL_R, CL_UNORM_INT16},
    {CL_R, CL_HALF_FLOAT},         {CL_R, CL_FLOAT},

    {CL_RG, CL_SIGNED_INT8},       {CL_RG, CL_SIGNED_INT16},
    {CL_RG, CL_SIGNED_INT32},      {CL_RG, CL_SNORM_INT8},
    {CL_RG, CL_SNORM_INT16},       {CL_RG, CL_UNSIGNED_INT8},
    {CL_RG, CL_UNSIGNED_INT16},    {CL_RG, CL_UNSIGNED_INT32},
    {CL_RG, CL_UNORM_INT8},        {CL_RG, CL_UNORM_INT16},
    {CL_RG, CL_HALF_FLOAT},        {CL_RG, CL_FLOAT},

    {CL_RGBA, CL_SIGNED_INT8},     {CL_RGBA, CL_SIGNED_INT16},
    {CL_RGBA, CL_SIGNED_INT32},    {CL_RGBA, CL_SNORM_INT8},
    {CL_RGBA, CL_SNORM_INT16},     {CL_RGBA, CL_UNSIGNED_INT8},
    {CL_RGBA, CL_UNSIGNED_INT16},  {CL_RGBA, CL_UNSIGNED_INT32},
    {CL_RGBA, CL_UNORM_INT8},      {CL_RGBA, CL_UNORM_INT16},
    {CL_RGBA, CL_HALF_FLOAT},      {CL_RGBA, CL_FLOAT},

    {CL_RGB, CL_UNORM_INT_101010}, {CL_RGB, CL_UNORM_SHORT_565},
    {CL_RGB, CL_UNORM_SHORT_555},
    //{CL_RGBA, CL_UNORM_SHORT_555},

};

static constexpr unsigned NumSupportedImageFormats =
    sizeof(SupportedImageFormats) / sizeof(SupportedImageFormats[0]);

static constexpr unsigned MaxPropertyEntries = 32;

static cl_device_unified_shared_memory_capabilities_intel
convertZeAllocCaps(ze_memory_access_cap_flags_t Flags) {
  cl_device_unified_shared_memory_capabilities_intel RetVal = 0;
  if (Flags & ZE_MEMORY_ACCESS_CAP_FLAG_RW)
    RetVal |= CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL;

  if (Flags & ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC)
    RetVal |= CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL;

  if (Flags & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT)
    RetVal |= CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL;

  if (Flags & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC)
    RetVal |= CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL;

  return RetVal;
}

Level0Device::Level0Device(Level0Driver *Drv, ze_device_handle_t DeviceH,
                           cl_device_id dev, const char *Parameters)
    : ClDev(dev), DeviceHandle(DeviceH), Driver(Drv),
      MemfillProgram(nullptr), ImagefillProgram(nullptr) {

  SETUP_DEVICE_CL_VERSION(dev, 3, 0);

  ClDev->available = &this->Available;
  ContextHandle = Drv->getContextHandle();
  assert(DeviceHandle);
  assert(ContextHandle);
  ze_result_t Res = ZE_RESULT_SUCCESS;
  HasGOffsets = Drv->hasExtension("ZE_experimental_global_offset");
  HasCompression = Drv->hasExtension("ZE_extension_memory_compression_hints");

  ze_device_properties_t DeviceProperties{};
  DeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
  DeviceProperties.pNext = nullptr;
  Res = zeDeviceGetProperties(DeviceHandle, &DeviceProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  ze_device_compute_properties_t ComputeProperties{};
  ComputeProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  ComputeProperties.pNext = nullptr;
  Res = zeDeviceGetComputeProperties(DeviceHandle, &ComputeProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  ze_device_module_properties_t ModuleProperties{};
  ModuleProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  ModuleProperties.pNext = nullptr;
  Res = zeDeviceGetModuleProperties(DeviceHandle, &ModuleProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  uint32_t QGroupPropCount = MaxPropertyEntries;
  ze_command_queue_group_properties_t QGroupProps[MaxPropertyEntries];
  for (uint32_t i = 0; i < MaxPropertyEntries; ++i) {
    QGroupProps[i].pNext = nullptr;
    QGroupProps[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  }
  Res = zeDeviceGetCommandQueueGroupProperties(DeviceHandle, &QGroupPropCount,
                                               QGroupProps);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  uint32_t MemPropCount = MaxPropertyEntries;
  ze_device_memory_properties_t MemProps[MaxPropertyEntries];
  for (uint32_t i = 0; i < MaxPropertyEntries; ++i) {
    MemProps[i].pNext = nullptr;
    MemProps[i].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  }
  Res = zeDeviceGetMemoryProperties(DeviceHandle, &MemPropCount, MemProps);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  ze_device_memory_access_properties_t MemAccessProperties{};
  MemAccessProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
  MemAccessProperties.pNext = nullptr;
  Res = zeDeviceGetMemoryAccessProperties(DeviceHandle, &MemAccessProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  uint32_t CachePropCount = MaxPropertyEntries;
  ze_device_cache_properties_t CacheProperties[MaxPropertyEntries];
  for (uint32_t i = 0; i < MaxPropertyEntries; ++i) {
    CacheProperties[i].pNext = nullptr;
    CacheProperties[i].stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
  }
  Res = zeDeviceGetCacheProperties(DeviceHandle, &CachePropCount,
                                   CacheProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  ze_device_image_properties_t ImageProperties{};
  ImageProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;
  ImageProperties.pNext = nullptr;
  Res = zeDeviceGetImageProperties(DeviceHandle, &ImageProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  ze_device_external_memory_properties_t ExternalMemProperties{};
  ExternalMemProperties.stype =
      ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
  ExternalMemProperties.pNext = nullptr;
  Res =
      zeDeviceGetExternalMemoryProperties(DeviceHandle, &ExternalMemProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    return;
  }

  POCL_MSG_PRINT_LEVEL0("all device info collected\n");

  // TODO
  ClDev->profiling_timer_resolution = 1;

  // Fixed props
  ClDev->compiler_available = CL_TRUE;
  ClDev->linker_available = CL_TRUE;
  ClDev->has_own_timer = CL_TRUE;
  ClDev->use_only_clang_opencl_headers = CL_TRUE;

  ClDev->local_as_id = SPIR_ADDRESS_SPACE_LOCAL;
  ClDev->constant_as_id = SPIR_ADDRESS_SPACE_CONSTANT;
  ClDev->global_as_id = SPIR_ADDRESS_SPACE_GLOBAL;

  // TODO the values here are copied from the Intel NEO.
  // we need a way to figure out the suitable values for
  // the real underlying device.
  ClDev->preferred_vector_width_char = 16;
  ClDev->preferred_vector_width_short = 8;
  ClDev->preferred_vector_width_int = 4;
  ClDev->preferred_vector_width_long = 1;
  ClDev->preferred_vector_width_float = 1;
  ClDev->preferred_vector_width_double = 1;
  ClDev->preferred_vector_width_half = 8;
  ClDev->native_vector_width_char = 16;
  ClDev->native_vector_width_short = 8;
  ClDev->native_vector_width_int = 4;
  ClDev->native_vector_width_long = 1;
  ClDev->native_vector_width_float = 1;
  ClDev->native_vector_width_double = 1;
  ClDev->native_vector_width_half = 8;

  ClDev->has_64bit_long = CL_TRUE;

  ClDev->endian_little = CL_TRUE;
  ClDev->parent_device = NULL;
  ClDev->max_sub_devices = 0;
  ClDev->num_partition_properties = 0;
  ClDev->partition_properties = NULL;
  ClDev->num_partition_types = 0;
  ClDev->partition_type = NULL;
  ClDev->max_constant_args = 8;
  ClDev->host_unified_memory = Integrated ? CL_TRUE : CL_FALSE;
  ClDev->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT;
  ClDev->global_var_max_size = 64 * 1024;

  ClDev->execution_capabilities = CL_EXEC_KERNEL;
  ClDev->address_bits = 64;
  ClDev->vendor = "Intel Corporation";
  ClDev->vendor_id = 0x8086;
  ClDev->profile = "FULL_PROFILE";

  ClDev->num_serialize_entries = 2;
  ClDev->serialize_entries = LEVEL0_SERIALIZE_ENTRIES;
  ClDev->llvm_cpu = nullptr;
  ClDev->llvm_target_triplet = "spir64-unknown-unknown";
  ClDev->generic_as_support = CL_TRUE;
  ClDev->supported_spir_v_versions = "SPIR-V_1.2";
  ClDev->on_host_queue_props = CL_QUEUE_PROFILING_ENABLE;
  ClDev->version_of_latest_passed_cts = "v2000-00-00-00";

  // deviceProperties
  switch (DeviceProperties.type) {
  case ZE_DEVICE_TYPE_CPU:
    ClDev->type = CL_DEVICE_TYPE_CPU;
    break;
  case ZE_DEVICE_TYPE_GPU:
    ClDev->type = CL_DEVICE_TYPE_GPU;
    break;
  default:
    ClDev->type = CL_DEVICE_TYPE_CUSTOM;
    // we don't know how to handle Custom devices yet
    // returning here leaves ClDev->available = CL_FALSE;
    return;
  }

  // ClDev->vendor_id = deviceProperties.vendorId;
  ClDev->short_name = ClDev->long_name = strdup(DeviceProperties.name);
  UUID = DeviceProperties.uuid;

  // ze_device_property_flags_t
  if ((DeviceProperties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0u) {
    Integrated = true;
  }
  if ((DeviceProperties.flags & ZE_DEVICE_PROPERTY_FLAG_ECC) != 0u) {
    ClDev->error_correction_support = CL_TRUE;
  }
  if ((DeviceProperties.flags & ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING) !=
      0u) {
    OndemandPaging = true;
  }

  ClDev->max_clock_frequency = DeviceProperties.coreClockRate;

  ClDev->max_mem_alloc_size = ClDev->max_constant_buffer_size =
      ClDev->global_var_pref_size = DeviceProperties.maxMemAllocSize;
  Supports64bitBuffers = (ClDev->max_mem_alloc_size > UINT32_MAX);

  MaxCommandQueuePriority = DeviceProperties.maxCommandQueuePriority;

  ClDev->max_compute_units = DeviceProperties.numSlices *
                             DeviceProperties.numSubslicesPerSlice *
                             DeviceProperties.numEUsPerSubslice;

  ClDev->preferred_wg_size_multiple =
      64; // deviceProperties.physicalEUSimdWidth;

  ClDev->profiling_timer_resolution = 1; // always one nanosecond

  /// When stype==::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES the
  ///< units are in nanoseconds. When
  ///< stype==::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 units are in
  ///< cycles/sec
  TimerFrequency = (double)DeviceProperties.timerResolution;
  TimerNsPerCycle = 1000000000.0 / TimerFrequency;

  TSBits = DeviceProperties.timestampValidBits;
  KernelTSBits = DeviceProperties.kernelTimestampValidBits;

#if 0
  /// support for subdevices. Currently unimplemented
  deviceProperties.subdeviceId  ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE
  uint32_t subDeviceCount = 0;
  zeDeviceGetSubDevices(device, &subDeviceCount, nullptr);
  ze_device_handle_t subDevices[2] = {};
  zeDeviceGetSubDevices(device, &subDeviceCount, subDevices);
#endif

  // computeProperties
  ClDev->max_work_group_size = ComputeProperties.maxTotalGroupSize;
  ClDev->max_work_item_dimensions = 3;
  ClDev->max_work_item_sizes[0] = ComputeProperties.maxGroupSizeX;
  ClDev->max_work_item_sizes[1] = ComputeProperties.maxGroupSizeY;
  ClDev->max_work_item_sizes[2] = ComputeProperties.maxGroupSizeZ;

  // level0 devices typically don't have unlimited number of groups per
  MaxWGCount[0] = ComputeProperties.maxGroupCountX;
  MaxWGCount[1] = ComputeProperties.maxGroupCountY;
  MaxWGCount[2] = ComputeProperties.maxGroupCountZ;
  POCL_MSG_PRINT_LEVEL0("Device Max WG counts: %u | %u | %u\n", MaxWGCount[0],
                        MaxWGCount[1], MaxWGCount[2]);

  ClDev->local_mem_type = CL_LOCAL;
  ClDev->local_mem_size = ComputeProperties.maxSharedLocalMemory;

  cl_uint Max = 0;
  if (ComputeProperties.numSubGroupSizes > 0) {
    for (unsigned i = 0; i < ComputeProperties.numSubGroupSizes; ++i) {
      if (ComputeProperties.subGroupSizes[i] > Max) {
        Max = ComputeProperties.subGroupSizes[i];
      }
    }
    ClDev->max_num_sub_groups = Max;

    SupportedSubgroupSizes.resize(ComputeProperties.numSubGroupSizes);
    for (unsigned i = 0; i < ComputeProperties.numSubGroupSizes; ++i) {
      SupportedSubgroupSizes[i] = ComputeProperties.subGroupSizes[i];
    }
  }

  // moduleProperties

  if ((ModuleProperties.flags & ZE_DEVICE_MODULE_FLAG_FP64) != 0u) {
    if ((ModuleProperties.fp64flags & ZE_DEVICE_FP_FLAG_DENORM) != 0u) {
      ClDev->double_fp_config |= CL_FP_DENORM;
    }
    if ((ModuleProperties.fp64flags & ZE_DEVICE_FP_FLAG_INF_NAN) != 0u) {
      ClDev->double_fp_config |= CL_FP_INF_NAN;
    }
    if ((ModuleProperties.fp64flags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST) !=
        0u) {
      ClDev->double_fp_config |= CL_FP_ROUND_TO_NEAREST;
    }
    if ((ModuleProperties.fp64flags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF) != 0u) {
      ClDev->double_fp_config |= CL_FP_ROUND_TO_INF;
    }
    if ((ModuleProperties.fp64flags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO) !=
        0u) {
      ClDev->double_fp_config |= CL_FP_ROUND_TO_ZERO;
    }
    if ((ModuleProperties.fp64flags & ZE_DEVICE_FP_FLAG_FMA) != 0u) {
      ClDev->double_fp_config |= CL_FP_FMA;
    }
    if ((ModuleProperties.fp64flags & ZE_DEVICE_FP_FLAG_SOFT_FLOAT) != 0u) {
      ClDev->double_fp_config |= CL_FP_SOFT_FLOAT;
    }
  } else {
    ClDev->double_fp_config = 0;
  }

  if ((ModuleProperties.flags & ZE_DEVICE_MODULE_FLAG_FP16) != 0u) {
    if ((ModuleProperties.fp16flags & ZE_DEVICE_FP_FLAG_DENORM) != 0u) {
      ClDev->half_fp_config |= CL_FP_DENORM;
    }
    if ((ModuleProperties.fp16flags & ZE_DEVICE_FP_FLAG_INF_NAN) != 0u) {
      ClDev->half_fp_config |= CL_FP_INF_NAN;
    }
    if ((ModuleProperties.fp16flags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST) !=
        0u) {
      ClDev->half_fp_config |= CL_FP_ROUND_TO_NEAREST;
    }
    if ((ModuleProperties.fp16flags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF) != 0u) {
      ClDev->half_fp_config |= CL_FP_ROUND_TO_INF;
    }
    if ((ModuleProperties.fp16flags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO) !=
        0u) {
      ClDev->half_fp_config |= CL_FP_ROUND_TO_ZERO;
    }
    if ((ModuleProperties.fp16flags & ZE_DEVICE_FP_FLAG_FMA) != 0u) {
      ClDev->half_fp_config |= CL_FP_FMA;
    }
    if ((ModuleProperties.fp16flags & ZE_DEVICE_FP_FLAG_SOFT_FLOAT) != 0u) {
      ClDev->half_fp_config |= CL_FP_SOFT_FLOAT;
    }
  } else {
    ClDev->half_fp_config = 0;
  }

  // single FP config
  {
    ClDev->single_fp_config = 0;
    if ((ModuleProperties.fp32flags & ZE_DEVICE_FP_FLAG_DENORM) != 0u) {
      ClDev->single_fp_config |= CL_FP_DENORM;
    }
    if ((ModuleProperties.fp32flags & ZE_DEVICE_FP_FLAG_INF_NAN) != 0u) {
      ClDev->single_fp_config |= CL_FP_INF_NAN;
    }
    if ((ModuleProperties.fp32flags & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST) !=
        0u) {
      ClDev->single_fp_config |= CL_FP_ROUND_TO_NEAREST;
    }
    if ((ModuleProperties.fp32flags & ZE_DEVICE_FP_FLAG_ROUND_TO_INF) != 0u) {
      ClDev->single_fp_config |= CL_FP_ROUND_TO_INF;
    }
    if ((ModuleProperties.fp32flags & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO) !=
        0u) {
      ClDev->single_fp_config |= CL_FP_ROUND_TO_ZERO;
    }
    if ((ModuleProperties.fp32flags & ZE_DEVICE_FP_FLAG_FMA) != 0u) {
      ClDev->single_fp_config |= CL_FP_FMA;
    }
    if ((ModuleProperties.fp32flags & ZE_DEVICE_FP_FLAG_SOFT_FLOAT) != 0u) {
      ClDev->single_fp_config |= CL_FP_SOFT_FLOAT;
    }
  }

  KernelUUID = ModuleProperties.nativeKernelSupported;

  std::string Extensions("cl_khr_byte_addressable_store"
                         " cl_khr_global_int32_base_atomics"
                         " cl_khr_global_int32_extended_atomics"
                         " cl_khr_local_int32_base_atomics"
                         " cl_khr_local_int32_extended_atomics"
                         " cl_khr_il_program"
                         " cl_khr_3d_image_writes");

  std::string OpenCL30Features("__opencl_c_images"
                               " __opencl_c_read_write_images"
                               " __opencl_c_3d_image_writes"
                               " __opencl_c_atomic_order_acq_rel"
                               " __opencl_c_atomic_order_seq_cst"
                               " __opencl_c_atomic_scope_device"
                               " __opencl_c_program_scope_global_variables"
                               " __opencl_c_generic_address_space");

  if ((ModuleProperties.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) != 0u) {
    Extensions.append(" cl_khr_int64_base_atomics"
                      " cl_khr_int64_extended_atomics");
  }
  if (ClDev->half_fp_config != 0u) {
    Extensions.append(" cl_khr_fp16");
    OpenCL30Features.append(" __opencl_c_fp16");
  }
  if (ClDev->double_fp_config != 0u) {
    Extensions.append(" cl_khr_fp64");
    OpenCL30Features.append(" __opencl_c_fp64");
  }
  if (ClDev->max_num_sub_groups > 0) {
    Extensions.append(" cl_khr_subgroups");
    OpenCL30Features.append(" __opencl_c_subgroups");
    OpenCL30Features.append(" __opencl_c_work_group_collective_functions");
  }
  if (ClDev->has_64bit_long != 0) {
    Extensions.append(" cl_khr_int64");
    OpenCL30Features.append(" __opencl_c_int64");
  }

  ClDev->device_side_printf = 0;
  ClDev->printf_buffer_size = ModuleProperties.printfBufferSize;
  ClDev->max_parameter_size = ModuleProperties.maxArgumentsSize;

  // memProps
  for (uint32_t i = 0; i < MemPropCount; ++i) {
    if (ClDev->global_mem_size < MemProps[i].totalSize) {
      ClDev->global_mem_size = MemProps[i].totalSize;
      GlobalMemOrd = i;
    }
  }
  if (ClDev->global_mem_size > ClDev->max_mem_alloc_size * 4) {
    ClDev->global_mem_size = ClDev->max_mem_alloc_size * 4;
  }

  // memAccessProperties
  if ((MemAccessProperties.sharedSingleDeviceAllocCapabilities &
       (ZE_MEMORY_ACCESS_CAP_FLAG_RW | ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC)) !=
      0u) {
    ClDev->svm_allocation_priority = 2;
    ClDev->atomic_memory_capabilities =
        CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_ORDER_ACQ_REL |
        CL_DEVICE_ATOMIC_ORDER_SEQ_CST | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP |
        CL_DEVICE_ATOMIC_SCOPE_DEVICE;
    ClDev->atomic_fence_capabilities =
        CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_ORDER_ACQ_REL |
        CL_DEVICE_ATOMIC_ORDER_SEQ_CST | CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM |
        CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP | CL_DEVICE_ATOMIC_SCOPE_DEVICE;
    // OpenCL 2.0 properties
    ClDev->svm_caps =
        CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS;
  } else {
    POCL_MSG_PRINT_LEVEL0("SVM disabled for device\n");
  }

  HostMemCaps = convertZeAllocCaps(MemAccessProperties.hostAllocCapabilities);
  DeviceMemCaps =
      convertZeAllocCaps(MemAccessProperties.deviceAllocCapabilities);
  SingleSharedCaps = convertZeAllocCaps(
      MemAccessProperties.sharedSingleDeviceAllocCapabilities);
  CrossSharedCaps = convertZeAllocCaps(
      MemAccessProperties.sharedCrossDeviceAllocCapabilities);
  SystemSharedCaps =
      convertZeAllocCaps(MemAccessProperties.sharedSystemAllocCapabilities);
  // the minimum capability required for USM
  if (DeviceMemCaps & ZE_MEMORY_ACCESS_CAP_FLAG_RW) {
    Extensions.append(" cl_intel_unified_shared_memory");
  }

  if (supportsDeviceUSM())
    Extensions.append(" cl_ext_buffer_device_address");

  ClDev->extensions = strdup(Extensions.c_str());
  ClDev->features = strdup(OpenCL30Features.c_str());

  pocl_setup_opencl_c_with_version(ClDev, CL_TRUE);
  pocl_setup_features_with_version(ClDev);
  pocl_setup_extensions_with_version(ClDev);
  pocl_setup_builtin_kernels_with_version(ClDev);
  pocl_setup_ils_with_version(ClDev);

  // cacheProperties
  for (uint32_t i = 0; i < CachePropCount; ++i) {
    // find largest cache that is not user-controlled
    if ((CacheProperties[i].flags &
         ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL) != 0u) {
      continue;
    }
    if (ClDev->global_mem_cache_size < CacheProperties[i].cacheSize) {
      ClDev->global_mem_cache_size = CacheProperties[i].cacheSize;
    }
  }
  ClDev->global_mem_cacheline_size = HOST_CPU_CACHELINE_SIZE;
  ClDev->global_mem_cache_type = CL_READ_WRITE_CACHE;

  // externalMemProperties
  ClDev->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT;

  // imageProperties
  ClDev->max_read_image_args = ImageProperties.maxReadImageArgs;
  ClDev->max_read_write_image_args = ImageProperties.maxWriteImageArgs;
  ClDev->max_write_image_args = ImageProperties.maxWriteImageArgs;
  ClDev->max_samplers = ImageProperties.maxSamplers;

  ClDev->image_max_array_size = ImageProperties.maxImageArraySlices;
  ClDev->image_max_buffer_size = ImageProperties.maxImageBufferSize;

  ClDev->image2d_max_height = ClDev->image2d_max_width =
      ImageProperties.maxImageDims2D;
  ClDev->image3d_max_depth = ClDev->image3d_max_height =
      ClDev->image3d_max_width = ImageProperties.maxImageDims3D;

  for (unsigned i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i) {
    ClDev->num_image_formats[i] = NumSupportedImageFormats;
    ClDev->image_formats[i] = SupportedImageFormats;
  }

  ClDev->image_support = CL_TRUE;

  // QGroupProps
  uint32_t UniversalQueueOrd = UINT32_MAX;
  uint32_t CopyQueueOrd = UINT32_MAX;
  uint32_t ComputeQueueOrd = UINT32_MAX;
  uint32_t NumUniversalQueues = 0;
  uint32_t NumCopyQueues = 0;
  uint32_t NumComputeQueues = 0;

  for (uint32_t i = 0; i < QGroupPropCount; ++i) {
    bool IsCompute = ((QGroupProps[i].flags &
                       ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0);
    bool IsCopy = ((QGroupProps[i].flags &
                    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0);
    if (IsCompute && IsCopy) {
      UniversalQueueOrd = i;
      NumUniversalQueues = QGroupProps[i].numQueues;
    }

    if (IsCompute && !IsCopy) {
      ComputeQueueOrd = i;
      NumComputeQueues = QGroupProps[i].numQueues;
    }

    if (!IsCompute && IsCopy) {
      CopyQueueOrd = i;
      NumCopyQueues = QGroupProps[i].numQueues;
    }
  }

  if (UniversalQueueOrd == UINT32_MAX &&
      (ComputeQueueOrd == UINT32_MAX || CopyQueueOrd == UINT32_MAX)) {
    POCL_MSG_ERR(
        "No universal queue and either of copy/compute queue are missing\n");
    return;
  }

  uint32_t_3 DeviceMaxWGs = {MaxWGCount[0], MaxWGCount[1], MaxWGCount[2]};

  uint64_t *CopyTimestamps = nullptr;
  uint64_t *ComputeTimestamps = nullptr;
  uint64_t *UniversalTimestamps = nullptr;

  // create specialized queues
  if (ComputeQueueOrd != UINT32_MAX && CopyQueueOrd != UINT32_MAX) {
    ComputeQueues.init(ComputeQueueOrd, NumComputeQueues, this);
    CopyQueues.init(CopyQueueOrd, NumCopyQueues, this);
  }

  // always create universal queues, if available
  if (UniversalQueueOrd != UINT32_MAX) {
    uint32_t num = std::max(1U, NumUniversalQueues);
    UniversalQueues.init(UniversalQueueOrd, num, this);
  }

  // calculate KernelCacheHash
  //
  // Note!!! there is no need to add Spec Constants or Compiler options
  // into KernelCacheHash, because pocl_cache_create_program_cachedir
  // has already taken care of those
  SHA1_CTX HashCtx;
  uint8_t Digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Init(&HashCtx);

  // not reliable
  // const ze_driver_uuid_t DrvUUID = Driver->getUUID();
  // pocl_SHA1_Update(&HashCtx, (const uint8_t*)&DrvUUID.id,
  // sizeof(DrvUUID.id));
  uint32_t DrvVersion = Driver->getVersion();
  pocl_SHA1_Update(&HashCtx, (const uint8_t *)&DrvVersion,
                   sizeof(DrvVersion));

  pocl_SHA1_Update(&HashCtx, (const uint8_t *)&DeviceProperties.type,
                   sizeof(DeviceProperties.type));

  pocl_SHA1_Update(&HashCtx, (const uint8_t *)&DeviceProperties.vendorId,
                   sizeof(DeviceProperties.vendorId));
  // not reliable
  // pocl_SHA1_Update(&HashCtx,
  //                 (const uint8_t*)&deviceProperties.uuid,
  //                 sizeof(deviceProperties.uuid));
  pocl_SHA1_Update(&HashCtx, (const uint8_t *)ClDev->short_name,
                   strlen(ClDev->short_name));
  pocl_SHA1_Final(&HashCtx, Digest);

  std::stringstream SStream;
  for (unsigned i = 0; i < sizeof(Digest); ++i) {
    SStream << std::setfill('0') << std::setw(2) << std::hex
            << (unsigned)Digest[i];
  }
  SStream.flush();
  KernelCacheHash = SStream.str();

  initHelperKernels();

  this->Available = CL_TRUE;
}

Level0Device::~Level0Device() {
  destroyHelperKernels();
  // ComputeQueues.wait()
  // CopyQueues.wait()
}

static void calculateHash(uint8_t *BuildHash,
                          const uint8_t *Data,
                          const size_t Len) {
  SHA1_CTX HashCtx;
  pocl_SHA1_Init(&HashCtx);
  pocl_SHA1_Update(&HashCtx, Data, Len);
  pocl_SHA1_Final(&HashCtx, BuildHash);
  uint8_t TempDigest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Final(&HashCtx, TempDigest);

  uint8_t *hashstr = BuildHash;
  for (unsigned i = 0; i < SHA1_DIGEST_SIZE; i++) {
    *hashstr++ = (TempDigest[i] & 0x0F) + 65;
    *hashstr++ = ((TempDigest[i] & 0xF0) >> 4) + 65;
  }
  *hashstr = 0;
  BuildHash[2] = '/';
}

bool Level0Device::initHelperKernels() {
  std::vector<uint8_t> SpvData;
  std::vector<char> ProgramBCData;
  std::string BuildLog;
  Level0Kernel *K;
  char ProgramCacheDir[POCL_MAX_PATHNAME_LENGTH];
  assert(Driver);

  // fake program with BuildHash to get a cache path
  struct _cl_program FakeProgram;
  FakeProgram.num_devices = 1;
  SHA1_digest_t BuildHash;
  FakeProgram.build_hash = &BuildHash;

  calculateHash(BuildHash, MemfillSpv, MemfillSpvLen);
  pocl_cache_program_path(ProgramCacheDir, &FakeProgram, 0);
  SpvData.clear();
  SpvData.insert(SpvData.end(), MemfillSpv, MemfillSpv + MemfillSpvLen);
  MemfillProgram = Driver->getJobSched().createProgram(
      ContextHandle, DeviceHandle,
      false, // JITCompilation,
      BuildLog,
      false, // Optimize,
      Supports64bitBuffers,
      0,       // SpecConstantIDs.size(),
      nullptr, // SpecConstantIDs.data(),
      nullptr, // SpecConstantPtrs.data(),
      nullptr, // SpecConstantSizes.data(),
      SpvData,
      ProgramBCData, // can be empty if JIT = disabled
      ProgramCacheDir, KernelCacheHash);
  if (MemfillProgram == nullptr) {
    POCL_MSG_ERR("Level0 Device: Failed to build memfill kernels");
    return false;
  }

  for (unsigned i = 1; i <= 128; i *= 2) {
    std::string Kernel1D = "memfill_" + std::to_string(i);
    K = Driver->getJobSched().createKernel(MemfillProgram, Kernel1D.c_str());
    assert(K);
    MemfillKernels[Kernel1D] = K;

    std::string Kernel3D = "memfill_rect_" + std::to_string(i);
    K = Driver->getJobSched().createKernel(MemfillProgram, Kernel1D.c_str());
    assert(K);
    MemfillKernels[Kernel3D] = K;
  }

  calculateHash(BuildHash, ImagefillSpv, ImagefillSpvLen);
  pocl_cache_program_path(ProgramCacheDir, &FakeProgram, 0);
  SpvData.clear();
  SpvData.insert(SpvData.end(), ImagefillSpv, ImagefillSpv + ImagefillSpvLen);
  ImagefillProgram = Driver->getJobSched().createProgram(
      ContextHandle, DeviceHandle,
      false, // JITCompilation,
      BuildLog,
      false, // Optimize,
      Supports64bitBuffers,
      0,       // SpecConstantIDs.size(),
      nullptr, // SpecConstantIDs.data(),
      nullptr, // SpecConstantPtrs.data(),
      nullptr, // SpecConstantSizes.data(),
      SpvData,
      ProgramBCData, // can be empty if JIT = disabled
      ProgramCacheDir, KernelCacheHash);
  if (ImagefillProgram == nullptr) {
    POCL_MSG_ERR("Level0 Device: Failed to build imagefill kernels");
    return false;
  }

  std::vector<std::string> PixelTypes = { "f", "ui", "i"};
  std::vector<std::string> ImgTypes = { "2d_", "2d_array_",
                                        "1d_", "1d_array_",
                                        "1d_buffer_",
                                        "3d_" };
  for (auto ImgT : ImgTypes) {
    for (auto PixT : PixelTypes) {
      std::string KernelName = "imagefill_" + ImgT + PixT;
      K = Driver->getJobSched().createKernel(ImagefillProgram,
                                             KernelName.c_str());
      assert(K);
      ImagefillKernels[KernelName] = K;
    }
  }

  return true;
}

void Level0Device::destroyHelperKernels() {
  if (MemfillProgram) {
    for (auto &I : MemfillKernels) {
      Driver->getJobSched().releaseKernel(MemfillProgram, I.second);
    }
    Driver->getJobSched().releaseProgram(MemfillProgram);
  }
  if (ImagefillProgram) {
    for (auto &I : ImagefillKernels) {
      Driver->getJobSched().releaseKernel(ImagefillProgram, I.second);
    }
    Driver->getJobSched().releaseProgram(ImagefillProgram);
  }
}

void Level0Device::pushCommand(_cl_command_node *Command) {
  if (Command->type == CL_COMMAND_NDRANGE_KERNEL ||
      Command->type == CL_COMMAND_SVM_MEMFILL ||
      Command->type == CL_COMMAND_MEMFILL_INTEL ||
      Command->type == CL_COMMAND_FILL_BUFFER ||
      Command->type == CL_COMMAND_FILL_IMAGE) {
    if (ComputeQueues.available())
      ComputeQueues.pushWork(Command);
    else
      UniversalQueues.pushWork(Command);
  } else {
    if (CopyQueues.available())
      CopyQueues.pushWork(Command);
    else
      UniversalQueues.pushWork(Command);
  }
}

void Level0Device::pushCommandBatch(BatchType Batch) {
  if (UniversalQueues.available())
    UniversalQueues.pushCommandBatch(Batch);
  else {
    POCL_ABORT_UNIMPLEMENTED("this code path should not be entered - BUG\n");
  }
}

void *Level0Device::allocSharedMem(uint64_t Size, bool EnableCompression,
                                   ze_device_mem_alloc_flags_t DevFlags,
                                   ze_host_mem_alloc_flags_t HostFlags) {
  void *Ptr = nullptr;
  ze_device_mem_alloc_desc_t MemAllocDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, DevFlags, GlobalMemOrd};
  ze_host_mem_alloc_desc_t HostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                       nullptr, HostFlags};

  ze_memory_compression_hints_ext_desc_t MemCompHints = {
      ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC, nullptr,
      ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED};
  if (EnableCompression && supportsCompression()) {
    MemAllocDesc.pNext = &MemCompHints;
  }

  uint64_t NextPowerOf2 = pocl_size_ceil2_64(Size);
  uint64_t Align = std::min(NextPowerOf2, (uint64_t)MAX_EXTENDED_ALIGNMENT);

  ze_result_t Res = zeMemAllocShared(ContextHandle, &MemAllocDesc, &HostDesc,
                                     Size, Align, DeviceHandle, &Ptr);
  LEVEL0_CHECK_RET(nullptr, Res);
  return Ptr;
}

void *Level0Device::allocDeviceMem(uint64_t Size,
                                   ze_device_mem_alloc_flags_t DevFlags) {
  void *Ptr = nullptr;
  ze_device_mem_alloc_desc_t MemAllocDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
      ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED, GlobalMemOrd};

  uint64_t NextPowerOf2 = pocl_size_ceil2_64(Size);
  uint64_t Align = std::min(NextPowerOf2, (uint64_t)MAX_EXTENDED_ALIGNMENT);

  ze_result_t Res = zeMemAllocDevice(ContextHandle, &MemAllocDesc, Size, Align,
                                     DeviceHandle, &Ptr);
  LEVEL0_CHECK_RET(nullptr, Res);
  return Ptr;
}

void *Level0Device::allocHostMem(uint64_t Size,
                                 ze_device_mem_alloc_flags_t HostFlags) {
  void *Ptr = nullptr;
  ze_host_mem_alloc_desc_t HostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                       nullptr, HostFlags};

  uint64_t NextPowerOf2 = pocl_size_ceil2_64(Size);
  uint64_t Align = std::min(NextPowerOf2, (uint64_t)MAX_EXTENDED_ALIGNMENT);

  ze_result_t Res = zeMemAllocHost(ContextHandle, &HostDesc, Size, Align, &Ptr);
  LEVEL0_CHECK_RET(nullptr, Res);
  return Ptr;
}

void Level0Device::freeMem(void *Ptr) {
  if (Ptr == nullptr)
    return;
  ze_result_t Res = zeMemFree(ContextHandle, Ptr);
  LEVEL0_CHECK_ABORT(Res);
}

bool Level0Device::freeMemBlocking(void *Ptr) {
  if (Ptr == nullptr)
    return true;

  if (!Driver->hasExtension("ZE_extension_memory_free_policies"))
    return false;

  ze_memory_free_ext_desc_t FreeExtDesc = {
      ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC, nullptr,
      ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE};
  ze_result_t Res = zeMemFreeExt(ContextHandle, &FreeExtDesc, Ptr);
  LEVEL0_CHECK_ABORT(Res);
  return true;
}

static void convertOpenclToZeImgFormat(cl_channel_type ChType,
                                       cl_channel_order ChOrder,
                                       ze_image_format_t &ZeFormat) {
  ze_image_format_type_t ZeType = {};
  ze_image_format_layout_t ZeLayout = {};

  switch (ChType) {
  case CL_SNORM_INT8:
  case CL_SNORM_INT16:
    ZeType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    break;
  case CL_UNORM_INT8:
  case CL_UNORM_INT16:
  case CL_UNORM_SHORT_555:
  case CL_UNORM_SHORT_565:
  case CL_UNORM_INT_101010:
    ZeType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    break;
  case CL_SIGNED_INT8:
  case CL_SIGNED_INT16:
  case CL_SIGNED_INT32:
    ZeType = ZE_IMAGE_FORMAT_TYPE_SINT;
    break;
  case CL_UNSIGNED_INT8:
  case CL_UNSIGNED_INT16:
  case CL_UNSIGNED_INT32:
    ZeType = ZE_IMAGE_FORMAT_TYPE_UINT;
    break;
  case CL_HALF_FLOAT:
  case CL_FLOAT:
    ZeType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    break;
  default:
    ZeType = ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32;
  }

  switch (ChOrder) {
  case CL_R: {
    ZeFormat.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
    ZeFormat.y = ZE_IMAGE_FORMAT_SWIZZLE_0;
    ZeFormat.z = ZE_IMAGE_FORMAT_SWIZZLE_0;
    ZeFormat.w = ZE_IMAGE_FORMAT_SWIZZLE_1;
    switch (ChType) {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_8;
      break;
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_16;
      break;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_32;
      break;
    default:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32;
    }
    break;
  }
  case CL_RG: {
    ZeFormat.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
    ZeFormat.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
    ZeFormat.z = ZE_IMAGE_FORMAT_SWIZZLE_0;
    ZeFormat.w = ZE_IMAGE_FORMAT_SWIZZLE_1;
    switch (ChType) {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8;
      break;
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16;
      break;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32;
      break;
    default:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32;
    }
    break;
  }
  case CL_RGB: {
    ZeFormat.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
    ZeFormat.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
    ZeFormat.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
    ZeFormat.w = ZE_IMAGE_FORMAT_SWIZZLE_1;
    switch (ChType) {
    case CL_UNORM_SHORT_565:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_5_6_5;
      break;
    case CL_UNORM_SHORT_555:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1;
      break;
    case CL_UNORM_INT_101010:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2;
      break;
    default:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32;
    }
    break;
  }
  case CL_RGBA: {
    ZeFormat.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
    ZeFormat.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
    ZeFormat.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
    ZeFormat.w = ZE_IMAGE_FORMAT_SWIZZLE_A;
    switch (ChType) {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
      break;
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
      break;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
      break;
    default:
      ZeLayout = ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32;
    }
    break;
  }
  }
  ZeFormat.layout = ZeLayout;
  ZeFormat.type = ZeType;
}

ze_image_handle_t Level0Device::allocImage(cl_channel_type ChType,
                                           cl_channel_order ChOrder,
                                           cl_mem_object_type ImgType,
                                           cl_mem_flags ImgFlags, size_t Width,
                                           size_t Height, size_t Depth) {

  // Specify single component FLOAT32 format
  ze_image_format_t ZeFormat{};
  convertOpenclToZeImgFormat(ChType, ChOrder, ZeFormat);
  ze_image_type_t ZeImgType;
  switch (ImgType) {
  case CL_MEM_OBJECT_IMAGE1D:
    ZeImgType = ZE_IMAGE_TYPE_1D;
    break;
  case CL_MEM_OBJECT_IMAGE2D:
    ZeImgType = ZE_IMAGE_TYPE_2D;
    break;
  case CL_MEM_OBJECT_IMAGE3D:
    ZeImgType = ZE_IMAGE_TYPE_3D;
    break;
  case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    ZeImgType = ZE_IMAGE_TYPE_1DARRAY;
    break;
  case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    ZeImgType = ZE_IMAGE_TYPE_2DARRAY;
    break;
  case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    ZeImgType = ZE_IMAGE_TYPE_BUFFER;
    break;
  default:
    ZeImgType = ZE_IMAGE_TYPE_FORCE_UINT32;
  }

  ze_image_flags_t ZeFlags = 0;
  if (((ImgFlags & CL_MEM_READ_WRITE) != 0u) ||
      ((ImgFlags & CL_MEM_WRITE_ONLY) != 0u)) {
    ZeFlags = ZE_IMAGE_FLAG_KERNEL_WRITE;
  }

  ze_image_desc_t imageDesc = {
      ZE_STRUCTURE_TYPE_IMAGE_DESC,
      nullptr,
      ZeFlags,
      ZeImgType,
      ZeFormat,
      (uint32_t)Width,
      (uint32_t)Height,
      (uint32_t)Depth,
      0, // array levels
      0  // mip levels
  };
  ze_image_handle_t ImageH = nullptr;
  ze_result_t Res =
      zeImageCreate(ContextHandle, DeviceHandle, &imageDesc, &ImageH);
  LEVEL0_CHECK_RET(nullptr, Res);
  return ImageH;
}

void Level0Device::freeImage(ze_image_handle_t ImageH) {
  ze_result_t Res = zeImageDestroy(ImageH);
  LEVEL0_CHECK_ABORT(Res);
}

ze_sampler_handle_t Level0Device::allocSampler(cl_addressing_mode AddrMode,
                                               cl_filter_mode FilterMode,
                                               cl_bool NormalizedCoords) {
  ze_sampler_address_mode_t ZeAddrMode = {};
  switch (AddrMode) {
  case CL_ADDRESS_NONE:
    ZeAddrMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
    break;
  case CL_ADDRESS_CLAMP:
    ZeAddrMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
    break;
  case CL_ADDRESS_CLAMP_TO_EDGE:
    ZeAddrMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    break;
  case CL_ADDRESS_REPEAT:
    ZeAddrMode = ZE_SAMPLER_ADDRESS_MODE_REPEAT;
    break;
  case CL_ADDRESS_MIRRORED_REPEAT:
    ZeAddrMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
    break;
  }

  ze_sampler_filter_mode_t ZeFilterMode = {};
  switch (FilterMode) {
  case CL_FILTER_LINEAR:
    ZeFilterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;
    break;
  case CL_FILTER_NEAREST:
    ZeFilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
    break;
  }

  ze_sampler_desc_t SamplerDesc = {
      ZE_STRUCTURE_TYPE_SAMPLER_DESC, nullptr, ZeAddrMode, ZeFilterMode,
      static_cast<ze_bool_t>((char)NormalizedCoords)};
  ze_sampler_handle_t SamplerH = nullptr;
  ze_result_t Res =
      zeSamplerCreate(ContextHandle, DeviceHandle, &SamplerDesc, &SamplerH);
  LEVEL0_CHECK_RET(nullptr, Res);
  return SamplerH;
}

void Level0Device::freeSampler(ze_sampler_handle_t SamplerH) {
  ze_result_t Res = zeSamplerDestroy(SamplerH);
  LEVEL0_CHECK_ABORT(Res);
}

int Level0Device::createProgram(cl_program Program, cl_uint DeviceI) {

  int Res = pocl_bitcode_is_spirv_execmodel_kernel(Program->program_il,
                                                   Program->program_il_size);
  POCL_RETURN_ERROR_ON((Res == 0), CL_BUILD_PROGRAM_FAILURE,
                       "Binary is not a SPIR-V module!\n");

  std::vector<uint8_t> Spirv;
  Spirv.resize(Program->program_il_size);
  for (size_t i = 0; i < Program->program_il_size; ++i) {
    Spirv[i] = static_cast<uint8_t>(Program->program_il[i]);
  }

  std::vector<char> ProgramBC;
  char *BinaryPtr = (char *)Program->binaries[DeviceI];
  size_t BinarySize = Program->binary_sizes[DeviceI];
  int TestR = pocl_bitcode_is_triple(BinaryPtr, BinarySize, "spir");
  assert(TestR && "Program->binaries[] is not LLVM bitcode!");
  ProgramBC.insert(ProgramBC.end(), BinaryPtr, BinaryPtr + BinarySize);

  assert(Program->data[DeviceI] == nullptr);
  char ProgramCacheDir[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_path(ProgramCacheDir, Program, DeviceI);

  std::vector<uint32_t> SpecConstantIDs;
  std::vector<const void *> SpecConstantPtrs;
  std::vector<size_t> SpecConstantSizes;

  if (Program->num_spec_consts != 0u) {
    for (size_t i = 0; i < Program->num_spec_consts; ++i) {
      if (Program->spec_const_is_set[i] == CL_FALSE) {
        continue;
      }
      SpecConstantIDs.push_back(Program->spec_const_ids[i]);
      SpecConstantPtrs.push_back(&Program->spec_const_values[i]);
      SpecConstantSizes.push_back(sizeof(uint64_t));
    }
  }

  std::string UserJITPref(pocl_get_string_option("POCL_LEVEL0_JIT", "auto"));
  bool JITCompilation = false;
  if (UserJITPref == "0")
    JITCompilation = false;
  else if (UserJITPref == "1")
    JITCompilation = true;
  else {
    // use heuristic
    if (UserJITPref != "auto")
      POCL_MSG_WARN("unknown option given to POCL_LEVEL0_JIT: '%s' \n",
                    UserJITPref.c_str());
    JITCompilation =
        (Program->num_kernels > 256 && Program->program_il_size > 128000);
  }
  POCL_MSG_PRINT_LEVEL0("createProgram | using JIT: %s\n",
                        (JITCompilation ? "YES" : "NO"));

  std::string CompilerOptions(
      Program->compiler_options != nullptr ? Program->compiler_options : "");
  bool Optimize =
      (CompilerOptions.find("-cl-disable-opt") == std::string::npos);

  std::string BuildLog;
  Level0Program *ProgramData = Driver->getJobSched().createProgram(
      ContextHandle, DeviceHandle, JITCompilation, BuildLog, Optimize,
      Supports64bitBuffers, SpecConstantIDs.size(), SpecConstantIDs.data(),
      SpecConstantPtrs.data(), SpecConstantSizes.data(), Spirv, ProgramBC,
      ProgramCacheDir, KernelCacheHash);

  if (ProgramData == nullptr) {
    if (!BuildLog.empty()) {
      pocl_append_to_buildlog(Program, DeviceI, strdup(BuildLog.c_str()),
                              BuildLog.size());
    }
    POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                         "Failed to compile program\n");
  }

  Program->data[DeviceI] = ProgramData;
  return CL_SUCCESS;
}

int Level0Device::freeProgram(cl_program Program, cl_uint DeviceI) {
  if (Program->data[DeviceI] == nullptr) {
    return CL_SUCCESS;
  }

  Level0Program *ProgramData = (Level0Program *)Program->data[DeviceI];
  Driver->getJobSched().releaseProgram(ProgramData);
  Program->data[DeviceI] = nullptr;
  return CL_SUCCESS;
}

bool Level0Device::getBestKernel(Level0Program *Program, Level0Kernel *Kernel,
                                 bool LargeOffset, unsigned LocalWGSize,
                                 ze_module_handle_t &Mod,
                                 ze_kernel_handle_t &Ker) {

  return Driver->getJobSched().getBestKernel(Program, Kernel, LargeOffset,
                                             LocalWGSize, Mod, Ker);
}

bool Level0Device::getMemfillKernel(unsigned PatternSize,
                                    Level0Kernel **L0Kernel,
                                    ze_module_handle_t &ModH,
                                    ze_kernel_handle_t &KerH) {

  std::string KernelName = "memfill_" + std::to_string(PatternSize);
  // TODO locking? errcheck!
  Level0Kernel *K = MemfillKernels[KernelName];
  assert(K);
  *L0Kernel = K;
  return Driver->getJobSched().getBestKernel(MemfillProgram, K,
                                             false, // LargeOffset,
                                             1024,  // LocalWGSize,
                                             ModH, KerH);
}

bool Level0Device::getImagefillKernel(cl_channel_type ChType,
                                      cl_channel_order ChOrder,
                                      cl_mem_object_type ImgType,
                                      Level0Kernel **L0Kernel,
                                      ze_module_handle_t &ModH,
                                      ze_kernel_handle_t &KerH) {

  std::string PixelType;
  switch (ChType) {
  case CL_UNSIGNED_INT8:
  case CL_UNSIGNED_INT16:
  case CL_UNSIGNED_INT32:
    PixelType = "ui";
    break;
  case CL_SIGNED_INT8:
  case CL_SIGNED_INT16:
  case CL_SIGNED_INT32:
    PixelType = "i";
    break;
  default:
    PixelType = "f";
  }
  std::string ImageType;
  switch (ImgType) {
  case CL_MEM_OBJECT_IMAGE1D:
    ImageType = "1d_";
    break;
  case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    ImageType = "1d_array_";
    break;
  case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    ImageType = "1d_buffer_";
    break;
  case CL_MEM_OBJECT_IMAGE2D:
    ImageType = "2d_";
    break;
  case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    ImageType = "2d_array_";
    break;
  case CL_MEM_OBJECT_IMAGE3D:
    ImageType = "3d_";
    break;
  default:
    ImageType = "_unknown";
    break;
  }

  std::string KernelName = "imagefill_" + ImageType + PixelType;
  // TODO locking? errcheck!
  Level0Kernel *K = ImagefillKernels[KernelName];
  assert(K);
  *L0Kernel = K;
  return Driver->getJobSched().getBestKernel(ImagefillProgram, K,
                                             false, // LargeOffset,
                                             1024,  // LocalWGSize,
                                             ModH, KerH);
}

cl_bitfield Level0Device::getMemCaps(cl_device_info Type) {
  switch (Type) {
  case CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL:
    return HostMemCaps;
  case CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL:
    return DeviceMemCaps;
  case CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
    return SingleSharedCaps;
  case CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
    return CrossSharedCaps;
  case CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL:
    return SystemSharedCaps;
  default:
    assert(0 && "unhandled switch value");
  }
  return 0;
}

void *Level0Device::getMemBasePtr(const void *USMPtr) {
  void *Base = nullptr;
  size_t Size = 0;
  ze_result_t Res = zeMemGetAddressRange(ContextHandle, USMPtr, &Base, &Size);
  if (Res != ZE_RESULT_SUCCESS)
    return nullptr;
  return Base;
}

size_t Level0Device::getMemSize(const void *USMPtr) {
  void *Base = nullptr;
  size_t Size = 0;
  ze_result_t Res = zeMemGetAddressRange(ContextHandle, USMPtr, &Base, &Size);
  if (Res != ZE_RESULT_SUCCESS)
    return 0;
  return Size;
}

cl_device_id Level0Device::getMemAssoc(const void *USMPtr) {
  ze_memory_allocation_properties_t Props = {};
  ze_device_handle_t AssocDev = nullptr;
  ze_result_t Res =
      zeMemGetAllocProperties(ContextHandle, USMPtr, &Props, &AssocDev);
  if (Res != ZE_RESULT_SUCCESS || AssocDev == nullptr)
    return nullptr;

  return Driver->getClDevForHandle(AssocDev);
}

cl_unified_shared_memory_type_intel
Level0Device::getMemType(const void *USMPtr) {
  ze_memory_allocation_properties_t Props = {};
  ze_device_handle_t AssocDev = nullptr;
  ze_result_t Res =
      zeMemGetAllocProperties(ContextHandle, USMPtr, &Props, &AssocDev);
  if (Res != ZE_RESULT_SUCCESS)
    return CL_MEM_TYPE_UNKNOWN_INTEL;

  switch (Props.type) {
  case ZE_MEMORY_TYPE_HOST:
    return CL_MEM_TYPE_HOST_INTEL;
  case ZE_MEMORY_TYPE_DEVICE:
    return CL_MEM_TYPE_DEVICE_INTEL;
  case ZE_MEMORY_TYPE_SHARED:
    return CL_MEM_TYPE_SHARED_INTEL;
  case ZE_MEMORY_TYPE_UNKNOWN:
  default:
    return CL_MEM_TYPE_UNKNOWN_INTEL;
  }
}

cl_mem_alloc_flags_intel Level0Device::getMemFlags(const void *USMPtr) {
  // TODO
  return 0;
}

void Level0Device::getTimingInfo(uint32_t &TS, uint32_t &KernelTS,
                                 double &TimerFreq, double &NsPerCycle) {
  TS = TSBits;
  KernelTS = KernelTSBits;
  TimerFreq = TimerFrequency;
  NsPerCycle = TimerNsPerCycle;
}

void Level0Device::getMaxWGs(uint32_t_3 *MaxWGs) {
  std::memcpy(MaxWGs, MaxWGCount, sizeof(uint32_t_3));
}

static constexpr unsigned MaxLevel0Devices = 1024;

Level0Driver::Level0Driver() {
  ze_result_t Res = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("zeInit FAILED\n");
    return;
  }
  uint32_t DriverCount = 1;
  Res = zeDriverGet(&DriverCount, &DriverH);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("zeDriverGet FAILED\n");
    return;
  }

  ze_driver_properties_t DriverProperties = {};
  DriverProperties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  DriverProperties.pNext = nullptr;
  Res = zeDriverGetProperties(DriverH, &DriverProperties);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("zeDriverGetProperties FAILED\n");
    return;
  }
  UUID = DriverProperties.uuid;
  Version = DriverProperties.driverVersion;

  uint32_t ExtCount = 0;
  Res = zeDriverGetExtensionProperties(DriverH, &ExtCount, nullptr);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("zeDriverGetExtensionProperties 1 FAILED\n");
    return;
  }

  std::vector<ze_driver_extension_properties_t> Extensions;
  if (ExtCount > 0) {
    POCL_MSG_PRINT_LEVEL0("%u Level0 extensions found\n", ExtCount);
    Extensions.resize(ExtCount);
    Res = zeDriverGetExtensionProperties(DriverH, &ExtCount, Extensions.data());
    if (Res != ZE_RESULT_SUCCESS) {
      POCL_MSG_ERR("zeDriverGetExtensionProperties 2 FAILED\n");
      return;
    }
    for (auto &E : Extensions) {
      POCL_MSG_PRINT_LEVEL0("Level0 extension: %s\n", E.name);
      ExtensionSet.insert(E.name);
    }
  } else {
    POCL_MSG_PRINT_LEVEL0("No Level0 extensions found\n");
  }

  ze_context_desc_t ContextDescription = {};
  ContextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ContextDescription.pNext = nullptr;
  ContextDescription.flags = 0;

  Res = zeContextCreate(DriverH, &ContextDescription, &ContextH);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("zeContextCreate FAILED\n");
    return;
  }

  uint32_t DeviceCount = 0;
  ze_device_handle_t DeviceArray[MaxLevel0Devices];
  Res = zeDeviceGet(DriverH, &DeviceCount, nullptr);
  if (Res != ZE_RESULT_SUCCESS || DeviceCount == 0
          || DeviceCount > MaxLevel0Devices) {
    POCL_MSG_ERR("zeDeviceGet 1 FAILED\n");
    return;
  }

  Res = zeDeviceGet(DriverH, &DeviceCount, DeviceArray);
  if (Res != ZE_RESULT_SUCCESS || DeviceCount == 0) {
    POCL_MSG_ERR("zeDeviceGet 2 FAILED\n");
    return;
  }

  Devices.resize(DeviceCount);
  DeviceHandles.resize(DeviceCount);
  for (uint32_t i = 0; i < DeviceCount; ++i) {
    DeviceHandles[i] = DeviceArray[i];
  }

  if (!JobSched.init(DriverH, DeviceHandles)) {
    Devices.clear();
    DeviceHandles.clear();
    POCL_MSG_ERR("Failed to initialize compilation job scheduler\n");
    return;
  }
}

Level0Driver::~Level0Driver() {
  Devices.clear();
  DeviceHandles.clear();
  if (ContextH != nullptr) {
    zeContextDestroy(ContextH);
  }
}

Level0Device *Level0Driver::createDevice(unsigned Index, cl_device_id Dev,
                                         const char *Params) {
  if (Index >= Devices.size()) {
    return nullptr;
  }
  assert(Devices[Index].get() == nullptr);
  Devices[Index].reset(
      new Level0Device(this, DeviceHandles[Index], Dev, Params));
  ++NumDevices;
  HandleToIDMap[DeviceHandles[Index]] = Dev;
  return Devices[Index].get();
}

void Level0Driver::releaseDevice(Level0Device *Dev) {
  if (empty()) {
    return;
  }
  for (auto &Device : Devices) {
    if (Device.get() == Dev) {
      Device.reset(nullptr);
      --NumDevices;
    }
  }
}
