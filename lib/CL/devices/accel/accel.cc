/* accel.cc - generic/example driver for hardware accelerators with memory
   mapped control.

   Copyright (c) 2019-2020 Pekka Jääskeläinen / Tampere University

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

#include "config.h"
#include "accel.h"
#include "Region.h"
#include "MMAPDevice.h"
#ifdef HAVE_XRT
#include "XrtDevice.h"
#endif
#include "EmulationDevice.h"
#include "TTASimDevice.h"
#include "accel-shared.h"
#include "almaif-compile-tce.h"
#include "almaif-compile.h"
#include "bufalloc.h"
#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl_cl.h"
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

//#define ACCEL_DUMP_MEMORY
pocl_lock_t globalMemIDLock = POCL_LOCK_INITIALIZER;
bool isGlobalMemIDSet = false;
int GlobalMemID;

pocl_lock_t runningLock = POCL_LOCK_INITIALIZER;
pocl_lock_t runningDeviceLock = POCL_LOCK_INITIALIZER;
int runningDeviceCount = 0;
pocl_thread_t runningThread;
void* runningThreadFunc(void*);
_cl_command_node *runningList;
bool runningJoinRequested = false;

BIKD::BIKD(BuiltinKernelId KernelIdentifier, const char *KernelName,
           const std::vector<pocl_argument_info> &ArgInfos)
    : KernelId(KernelIdentifier) {

  builtin_kernel = 1;
  name = strdup(KernelName);
  num_args = ArgInfos.size();
  arg_info = new pocl_argument_info[num_args];
  int i = 0;
  for (auto ArgInfo : ArgInfos) {
    arg_info[i] = ArgInfo;
    arg_info[i].name = strdup(ArgInfo.name);
    arg_info[i].type_name = strdup(ArgInfo.type_name);
    ++i;
  }
}

// Shortcut handles to make the descriptor list more compact.
#define PTR_ARG POCL_ARG_TYPE_POINTER
#define GLOBAL_AS CL_KERNEL_ARG_ADDRESS_GLOBAL

BIKD BIDescriptors[] = {BIKD(POCL_CDBI_COPY, "pocl.copy",
                             {BIArg("char*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("char*", "output", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_ADD32, "pocl.add32",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_MUL32, "pocl.mul32",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_LEDBLINK, "pocl.ledblink",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "input", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_COUNTRED, "pocl.countred",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)})};



void pocl_accel_init_device_ops(struct pocl_device_ops *ops) {

  ops->device_name = "accel";
  ops->init = pocl_accel_init;
  ops->uninit = pocl_accel_uninit;
  ops->probe = pocl_accel_probe;
  ops->build_hash = pocl_accel_build_hash;
  ops->setup_metadata = pocl_accel_setup_metadata;

  /* TODO: Bufalloc-based allocation from the onchip memories. */
  ops->alloc_mem_obj = pocl_accel_alloc_mem_obj;
  ops->free = pocl_accel_free;
  ops->map_mem = pocl_accel_map_mem;
  ops->unmap_mem = pocl_accel_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->submit = pocl_accel_submit;
  ops->join = pocl_accel_join;

  ops->write = pocl_accel_write;
  ops->read = pocl_accel_read;

  ops->notify = pocl_accel_notify;

  ops->broadcast = pocl_broadcast;
  ops->run = pocl_accel_run;

//  ops->update_event = pocl_accel_update_event;
  ops->free_event_data = pocl_accel_free_event_data;
  ops->wait_event = pocl_accel_wait_event;



#if 0
    ops->read_rect = pocl_accel_read_rect;
    ops->write_rect = pocl_accel_write_rect;
    ops->unmap_mem = pocl_accel_unmap_mem;
    ops->memfill = pocl_accel_memfill;
    ops->copy = pocl_accel_copy;

    // new driver api (out-of-order): TODO implement these for accel
    ops->wait_event = pocl_accel_wait_event;
    ops->free_event_data = pocl_accel_free_event_data;
    ops->init_target_machine = NULL;
    ops->init_build = pocl_accel_init_build;
#endif
}

void pocl_accel_write(void *data, const void *__restrict__ src_host_ptr,
                      pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                      size_t offset, size_t size) {
  chunk_info_t *chunk = (chunk_info_t *)dst_mem_id->mem_ptr;
  size_t dst = chunk->start_address + offset;
  AccelData *d = (AccelData *)data;
  POCL_MSG_PRINT_INFO("accel: Copying 0x%zu bytes to 0x%zu\n", size, dst);
  d->Dev->DataMemory->CopyToMMAP(dst, src_host_ptr, size);
}

void pocl_accel_read(void *data, void *__restrict__ dst_host_ptr,
                     pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                     size_t offset, size_t size) {
  chunk_info_t *chunk = (chunk_info_t *)src_mem_id->mem_ptr;
  size_t src = chunk->start_address + offset;
  AccelData *d = (AccelData *)data;
  POCL_MSG_PRINT_INFO("accel: Copying 0x%zu bytes from 0x%zu\n", size, src);
  d->Dev->DataMemory->CopyFromMMAP(dst_host_ptr, src, size);
}

cl_int pocl_accel_alloc_mem_obj(cl_device_id device, cl_mem mem_obj,
                                void *host_ptr) {

  AccelData *data = (AccelData *)device->data;
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];
  assert(p->mem_ptr == NULL);
  chunk_info_t *chunk = NULL;

  /* accel driver doesn't preallocate */
  if ((mem_obj->flags & CL_MEM_ALLOC_HOST_PTR) && (mem_obj->mem_host_ptr == NULL))
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  chunk = pocl_alloc_buffer_from_region(&data->Dev->AllocRegion, mem_obj->size);
  if (chunk == NULL)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  POCL_MSG_PRINT_MEMORY ("accel: allocated 0x%zu bytes from 0x%zu\n",
                          mem_obj->size, chunk->start_address);

  p->mem_ptr = chunk;
  p->version = 0;

  return CL_SUCCESS;
}


void pocl_accel_free(cl_device_id device, cl_mem mem) {

  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  AccelData *data = (AccelData *)device->data;

  chunk_info_t *chunk =
      (chunk_info_t *)p->mem_ptr;

  POCL_MSG_PRINT_MEMORY ("accel: freed 0x%zu bytes from 0x%zu\n",
                          mem->size, chunk->start_address);

  assert(chunk != NULL);
  pocl_free_chunk(chunk);

  p->mem_ptr = NULL;
  p->version = 0;

}

cl_int pocl_accel_map_mem(void *data, pocl_mem_identifier *src_mem_id,
                          cl_mem src_buf, mem_mapping_t *map) {

  /* Synch the device global region to the host memory. */
  pocl_accel_read(data, map->host_ptr, src_mem_id, src_buf, map->offset,
                  map->size);

  return CL_SUCCESS;
}

cl_int pocl_accel_unmap_mem(void *data, pocl_mem_identifier *dst_mem_id,
                            cl_mem dst_buf, mem_mapping_t *map) {

  if (map->map_flags != CL_MAP_READ) {
  /* Synch the host memory to the device global region. */
  pocl_accel_write(data, map->host_ptr, dst_mem_id, dst_buf, map->offset,
                  map->size);
  }

  return CL_SUCCESS;
}


cl_int pocl_accel_init(unsigned j, cl_device_id dev, const char *parameters) {

  SETUP_DEVICE_CL_VERSION(1, 2);
  dev->type = CL_DEVICE_TYPE_CUSTOM;
  dev->long_name = (char *)"memory mapped custom device";
  dev->vendor = "pocl";
  dev->version = "1.2";
  dev->available = CL_TRUE;
  dev->extensions = "";
  dev->profile = "FULL_PROFILE";
  dev->max_mem_alloc_size = 100 * 1024 * 1024;
  dev->mem_base_addr_align = 16;

  AccelData *D = new AccelData;
  dev->data = (void *)D;

  if (!parameters) {
    POCL_ABORT("accel: parameters were not given\n");
  }

  // strtok_r() modifies string, copy it to be nice
  char *scanParams = strdup(parameters);

  char *savePtr;
  char *paramToken = strtok_r(scanParams, ",", &savePtr);
  D->BaseAddress = strtoull(paramToken, NULL, 0);

  std::string supportedList;
  char xrt_kernel_name[100];
  if (D->BaseAddress != 0xE) {
    paramToken = strtok_r(NULL, ",", &savePtr);
    strcpy(xrt_kernel_name, paramToken);
    POCL_MSG_PRINT_INFO("accel: enabling device with device kernel name %s",
                        xrt_kernel_name);
  }

  bool enable_compilation = false;

  while (paramToken = strtok_r(NULL, ",", &savePtr)) {
    auto token = strtoul(paramToken, NULL, 0);
    BuiltinKernelId kernelId = static_cast<BuiltinKernelId>(token);
    size_t numBIKDs = sizeof(BIDescriptors) / sizeof(*BIDescriptors);

    bool found = false;
    for (size_t i = 0; i < numBIKDs; ++i) {
      if (BIDescriptors[i].KernelId == kernelId) {
        if (supportedList.size() > 0)
          supportedList += ";";
        supportedList += BIDescriptors[i].name;
        D->SupportedKernels.insert(&BIDescriptors[i]);
        found = true;
        break;
      }
    }
    if (kernelId == POCL_CDBI_JIT_COMPILER) {
      enable_compilation = true;
    } else if (!found) {
      POCL_ABORT("accel: Unknown Kernel ID (%lu) given\n", token);
    }
  }

  dev->builtin_kernel_list = strdup(supportedList.c_str());
  if (enable_compilation) {
    POCL_MSG_PRINT_INFO("Enabling compilation\n");
    dev->ops->create_kernel = pocl_almaif_create_kernel;
    dev->ops->free_kernel = pocl_almaif_free_kernel;
    dev->ops->compile_kernel = pocl_almaif_compile_kernel;
    dev->ops->build_hash = pocl_almaif_build_hash;
    dev->ops->build_poclbinary = pocl_driver_build_poclbinary;
    dev->ops->build_binary = pocl_almaif_build_binary;

    dev->ops->build_source = pocl_driver_build_source;
    dev->ops->link_program = pocl_driver_link_program;
    dev->ops->free_program = pocl_driver_free_program;

    dev->ops->setup_metadata = pocl_driver_setup_metadata;

    dev->compiler_available = CL_TRUE;
    dev->linker_available = CL_TRUE;

  } else {
    D->compilationData = NULL;
    dev->compiler_available = CL_FALSE;
    dev->linker_available = CL_FALSE;
  }

  if(!pocl_offline_compile){

    POCL_MSG_PRINT_INFO("accel: accelerator at 0x%zx with %zu builtin kernels (%s)\n",
        D->BaseAddress, D->SupportedKernels.size(), dev->builtin_kernel_list);
    // Recognize whether we are emulating or not
    if (D->BaseAddress == EMULATING_ADDRESS) {
      D->Dev = new EmulationDevice();
#ifdef HAVE_XRT
    } else if (D->BaseAddress == 0xA) {
      D->Dev = new XrtDevice(xrt_kernel_name);
#endif
    } else if (D->BaseAddress == 0xB) {
      D->Dev = new TTASimDevice(xrt_kernel_name);
    } else {
      D->Dev = new MMAPDevice(D->BaseAddress, xrt_kernel_name);
    }

    if (!(D->Dev->RelativeAddressing))
    {
      POCL_LOCK(globalMemIDLock);
      if (isGlobalMemIDSet)
      {
        dev->global_mem_id = GlobalMemID;
      }
      else
      {
        GlobalMemID = dev->dev_id;
        isGlobalMemIDSet = true;
      }
      POCL_UNLOCK(globalMemIDLock);
    }
    dev->global_mem_size = D->Dev->DataMemory->Size;
  } else {
    POCL_MSG_PRINT_INFO("Starting offline compilation device initialization\n");
  }
  if (enable_compilation) {
    char adf_file[200];
    snprintf(adf_file, 200, "%s.adf", xrt_kernel_name);
    pocl_almaif_init(j, dev, adf_file);
  }

  POCL_MSG_PRINT_INFO("accel: mmap done\n");
  if (pocl_offline_compile){
    std::cout <<"Offline compilation device initialized"<<std::endl;
    return CL_SUCCESS;
  }
  for (int i = 0; i < (D->Dev->DataMemory->Size >> 2); i++) {
//    D->Dev->DataMemory->Write32(4 * i, 0);
  }
  for (int i = 0; i < (D->Dev->CQMemory->Size >> 2); i++) {
//    D->Dev->CQMemory->Write32(4 * i, 0);
  }
  // Initialize AQL queue by setting all headers to invalid
  for (uint32_t i = AQL_PACKET_LENGTH; i < D->Dev->CQMemory->Size; i += AQL_PACKET_LENGTH) {
    POCL_MSG_PRINT_INFO("Initializing AQL Packet cqmemory size=%i\n",D->Dev->CQMemory->Size);
    D->Dev->CQMemory->Write16(i, AQL_PACKET_INVALID);
  }


#ifdef ACCEL_DUMP_MEMORY
  POCL_MSG_PRINT_INFO("INIT MEMORY DUMP\n");
  D->Dev->printMemoryDump();
#endif

  // Lift accelerator reset
  D->Dev->ControlMemory->Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_CONTINUE_CMD);

  // This would be more logical as a per builtin kernel value?
  // Either way, the minimum is 3 for a device
  dev->max_work_item_dimensions = 3;
  dev->max_work_group_size = dev->max_work_item_sizes[0] =
      dev->max_work_item_sizes[1] = dev->max_work_item_sizes[2] = 1024;

  D->ReadyList = NULL;
  D->CommandList = NULL;
  POCL_INIT_LOCK(D->CommandListLock);
  POCL_INIT_LOCK(D->AQLQueueLock);

  POCL_LOCK(runningDeviceLock);
  if(runningDeviceCount == 0) {
    runningJoinRequested = false;
    POCL_CREATE_THREAD(runningThread, &runningThreadFunc, NULL);
  }
  runningDeviceCount++;
  POCL_UNLOCK(runningDeviceLock);


  if (D->BaseAddress == EMULATING_ADDRESS) {
    std::cout << "Custom emulation device " << j << " initialized" << std::endl;
  } else {
    std::cout << "Custom device " << j << " initialized" << std::endl;
  }
  return CL_SUCCESS;
}

cl_int pocl_accel_uninit(unsigned j, cl_device_id device) {
  POCL_MSG_PRINT_INFO("accel: uninit\n");

  POCL_LOCK(runningDeviceLock);
  runningDeviceCount--;
  if(runningDeviceCount == 0) {
    runningJoinRequested = true;
    POCL_JOIN_THREAD(runningThread);
  }
  POCL_UNLOCK(runningDeviceLock);


  AccelData *D = (AccelData *)device->data;
  if (D->compilationData != NULL) {
    pocl_almaif_uninit(j, device);
    D->compilationData = NULL;
  }

  delete D->Dev;
  delete D;
  return CL_SUCCESS;
}

unsigned int pocl_accel_probe(struct pocl_device_ops *ops) {
  int env_count = pocl_device_get_env_count(ops->device_name);
  return env_count;
}

char *pocl_accel_build_hash(cl_device_id /*device*/) {
  char *res = (char *)calloc(1000, sizeof(char));
  snprintf(res, 1000, "accel-%s", HOST_DEVICE_BUILD_HASH);
  return res;
}

static cl_int
pocl_accel_get_builtin_kernel_metadata(void *data, const char *kernel_name,
                                       pocl_kernel_metadata_t *target) {
  AccelData *D = (AccelData *)data;
  BIKD *Desc = nullptr;
  for (size_t i = 0; i < sizeof(BIDescriptors) / sizeof(BIDescriptors[0]);
       ++i) {
    Desc = &BIDescriptors[i];
    if (std::string(Desc->name) == kernel_name) {
      memcpy(target, (pocl_kernel_metadata_t *)Desc,
             sizeof(pocl_kernel_metadata_t));
      target->name = strdup(Desc->name);
      target->arg_info = (struct pocl_argument_info *)calloc(
          Desc->num_args, sizeof(struct pocl_argument_info));
      memset(target->arg_info, 0,
             sizeof(struct pocl_argument_info) * Desc->num_args);
      for (unsigned Arg = 0; Arg < Desc->num_args; ++Arg) {
        memcpy(&target->arg_info[Arg], &Desc->arg_info[Arg],
               sizeof(pocl_argument_info));
        target->arg_info[Arg].name = strdup(Desc->arg_info[Arg].name);
        target->arg_info[Arg].type_name = strdup(Desc->arg_info[Arg].type_name);
      }
    }
  }
  return 0;
}

int pocl_accel_setup_metadata(cl_device_id device, cl_program program,
                              unsigned program_device_i) {
  if (program->builtin_kernel_names == nullptr)
    return 0;

  program->num_kernels = program->num_builtin_kernels;
  if (program->num_kernels) {
    program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
        program->num_kernels, sizeof(pocl_kernel_metadata_t));

    for (size_t i = 0; i < program->num_kernels; ++i) {
      pocl_accel_get_builtin_kernel_metadata(device->data,
                                             program->builtin_kernel_names[i],
                                             &program->kernel_meta[i]);
    }
  }

  return 1;
}

static void scheduleCommands(AccelData &D) {

  _cl_command_node *Node;
  // Execute commands from ready list.
  while ((Node = D.ReadyList)) {
    assert(pocl_command_is_ready(Node->event));
    assert(Node->event->status == CL_SUBMITTED);

    POCL_UNLOCK(D.CommandListLock);

    pocl_exec_command(Node);
    POCL_LOCK(D.CommandListLock);
    CDL_DELETE(D.ReadyList, Node);
  }

  return;
}

void pocl_accel_submit(_cl_command_node *Node, cl_command_queue /*CQ*/) {

  Node->ready = 1;

  struct AccelData *D = (AccelData *)Node->device->data;
  
  if (Node->type == CL_COMMAND_NDRANGE_KERNEL) {
    pocl_update_event_submitted(Node->event);

    POCL_UNLOCK_OBJ(Node->event);
    pocl_update_event_running(Node->event);

    submit_and_barrier((AccelData*)Node->device->data, Node);
    submit_kernel_packet((AccelData*)Node->device->data, Node);

    POCL_LOCK(runningLock);
    DL_PREPEND(runningList, Node);
    POCL_UNLOCK(runningLock);
  }
  else {
    POCL_LOCK(D->CommandListLock);
    pocl_command_push(Node, &D->ReadyList, &D->CommandList);

    POCL_UNLOCK_OBJ(Node->event);
    scheduleCommands(*D);
    POCL_UNLOCK(D->CommandListLock);
  }

  return;
}

void pocl_accel_join(cl_device_id Device, cl_command_queue /*CQ*/) {

  struct AccelData *D = (AccelData *)Device->data;
  POCL_LOCK(D->CommandListLock);
  while(D->CommandList || D->ReadyList) {
    //scheduleCommands(*D);
 
    POCL_UNLOCK(D->CommandListLock);
    usleep(1000);
    POCL_LOCK(D->CommandListLock);
  }

  //scheduleCommands(*D);
  POCL_UNLOCK(D->CommandListLock);

  _cl_command_node* Node;
  bool stillRunning = true;
  while (stillRunning) {
    stillRunning = false;
    POCL_LOCK(runningLock);
    if(runningList) {
      DL_FOREACH(runningList, Node)
      {
        if (Node->device == Device) {
          stillRunning = true;
          break;
        }
      }
    }
    POCL_UNLOCK(runningLock);
    if(stillRunning) {
      usleep(1000);
    }
  }
  return;
}

void pocl_accel_notify(cl_device_id Device, cl_event Event, cl_event Finished) {

  struct AccelData &D = *(AccelData *)Device->data;

  _cl_command_node *volatile Node = Event->command;

  if (Finished->status < CL_COMPLETE) {
    pocl_update_event_failed(Event);
    return;
  }

  if (!Node->ready)
    return;

  if (Event->command->type != CL_COMMAND_NDRANGE_KERNEL) {
  if (pocl_command_is_ready(Event)) {
    if (Event->status == CL_QUEUED) {
      pocl_update_event_submitted(Event);
      POCL_LOCK(D.CommandListLock);
      CDL_DELETE(D.CommandList, Node);
      CDL_PREPEND(D.ReadyList, Node);

      POCL_UNLOCK_OBJ(Node->event);
      scheduleCommands(D);
      POCL_LOCK_OBJ(Node->event);
      POCL_UNLOCK(D.CommandListLock);
    }
    return;
  }
  }
}

void scheduleNDRange(AccelData *data, _cl_command_node *cmd,
                              size_t arg_size, void *arguments) {
  _cl_command_run *run = &cmd->command.run;
  int32_t kernelID = -1;
  for (auto supportedKernel : data->SupportedKernels) {
    if (strcmp(supportedKernel->name, run->kernel->name) == 0)
      kernelID = (int32_t)supportedKernel->KernelId;
  }

  if (kernelID == -1) {
    if (data->compilationData == NULL) {
      POCL_ABORT("accel: scheduled an NDRange with unsupported kernel\n");
    } else {
      POCL_MSG_PRINT_INFO(
          "accel: fixed function kernel not found, start compiling:\n");
      pocl_almaif_compile_kernel(cmd, run->kernel, cmd->device, 1);
    }
  }

  // Additional space for a signal
  size_t extraAlloc = sizeof(uint32_t);
  chunk_info_t *chunk =
     pocl_alloc_buffer_from_region(&data->Dev->AllocRegion, arg_size + extraAlloc);
  assert(chunk && "Failed to allocate signal/argument buffer");

  POCL_MSG_PRINT_INFO("accel: allocated 0x%zx bytes for signal/arguments "
                      "from 0x%zx\n",
                      arg_size + extraAlloc, chunk->start_address);
  cmd->event->data = (void *) chunk;

  size_t signalAddress = chunk->start_address;
  size_t argsAddress = chunk->start_address + sizeof(uint32_t);
  POCL_MSG_PRINT_INFO("Signal address=0x%zx\n", signalAddress);
  // Set initial signal value
  data->Dev->DataMemory->Write32(signalAddress - data->Dev->DataMemory->PhysAddress, 0);
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
    size_t pc_start_addr = data->compilationData->pocl_context->start_address;
    data->Dev->DataMemory->CopyToMMAP(pc_start_addr, &pc, sizeof(pocl_context32));

    if (data->Dev->RelativeAddressing) {
      pc_start_addr -= data->Dev->DataMemory->PhysAddress;
    }

    packet.reserved = pc_start_addr;

    almaif_kernel_data_t *kd = (almaif_kernel_data_t *)run->kernel->data[cmd->program_device_i];
    packet.kernel_object = kd->kernel_address;

    POCL_MSG_PRINT_INFO("Kernel addresss=%zu\n", kd->kernel_address);
  }

  if (data->Dev->RelativeAddressing) {
    packet.kernarg_address = argsAddress - data->Dev->DataMemory->PhysAddress;
    packet.completion_signal = signalAddress - data->Dev->DataMemory->PhysAddress;
  } else {
    packet.kernarg_address = argsAddress;
    packet.completion_signal = signalAddress;
  }

  POCL_MSG_PRINT_INFO("ArgsAddress=%llu SignalAddress=%llu\n",
                      packet.kernarg_address, packet.completion_signal);

  POCL_LOCK(data->AQLQueueLock);
  uint32_t queue_length = data->Dev->CQMemory->Size / AQL_PACKET_LENGTH - 1;

  uint32_t write_iter = data->Dev->CQMemory->Read32(ACCEL_CQ_WRITE);
  uint32_t read_iter = data->Dev->CQMemory->Read32(ACCEL_CQ_READ);
  while (write_iter >= read_iter + queue_length) {
    //POCL_MSG_PRINT_INFO("write_iter=%u, read_iter=%u length=%u", write_iter, read_iter, queue_length);
    read_iter = data->Dev->CQMemory->Read32(ACCEL_CQ_READ);
  }
  uint32_t packet_loc = (write_iter & (queue_length - 1)) * AQL_PACKET_LENGTH + AQL_PACKET_LENGTH;
  data->Dev->CQMemory->CopyToMMAP(packet_loc + data->Dev->CQMemory->PhysAddress,
                                   &packet, 64);


#ifdef ACCEL_DUMP_MEMORY
  POCL_MSG_PRINT_INFO("PRELAUNCH MEMORY DUMP\n");
  data->Dev->printMemoryDump();
#endif

  // finally, set header as not-invalid
  data->Dev->CQMemory->Write16(packet_loc,
                          (1<<AQL_PACKET_KERNEL_DISPATCH) | AQL_PACKET_BARRIER);

  POCL_MSG_PRINT_INFO(
      "accel: Handed off a packet for execution, write iter=%u\n", write_iter);
  // Increment queue index
  data->Dev->CQMemory->Write32(ACCEL_CQ_WRITE, write_iter + 1);

  POCL_UNLOCK(data->AQLQueueLock);
}

bool isEventDone(AccelData* data, cl_event event) {

  size_t offset = ((chunk_info_t*)event->data)->start_address - data->Dev->DataMemory->PhysAddress;

  uint32_t status = data->Dev->DataMemory->Read32(offset);

  return (status != 0);
}

void pocl_accel_wait_event(cl_device_id device, cl_event event)
{
  AccelData *D = (AccelData *)(device->data);
  size_t offset = ((chunk_info_t *)event->data)->start_address - D->Dev->DataMemory->PhysAddress;

  POCL_MSG_PRINT_INFO("accel wait on event: ADDRESS=%d, PHYSDDRESS=%d, OFFSET=%d\n", ((chunk_info_t *)event->data)->start_address, D->Dev->DataMemory->PhysAddress, offset);

  uint32_t status;
  int counter = 1;
  do
  {
    usleep(200);
    status = D->Dev->DataMemory->Read32(offset);

#ifdef ACCEL_DUMP_MEMORY
    uint64_t cyclecount = D->Dev->ControlMemory->Read64(ACCEL_STATUS_REG_CC_LOW);
    uint64_t stallcount = D->Dev->ControlMemory->Read64(ACCEL_STATUS_REG_SC_LOW);
    uint32_t programcounter = D->Dev->ControlMemory->Read32(ACCEL_STATUS_REG_PC);
    POCL_MSG_PRINT_INFO(
        "Accel: RUNNING Cyclecount=%lu Stallcount=%lu Current PC=%u\n",
        cyclecount, stallcount, programcounter);

    if (counter % 2000 == 0)
    {
      D->Dev->printMemoryDump();
    }
    counter++;

#endif

  } while (status == 0);
}

void submit_and_barrier(AccelData *D, _cl_command_node *cmd){

  event_node* dep_event = cmd->event->wait_list;
  if (dep_event == NULL){
    POCL_MSG_PRINT_INFO("Accel: no events to wait for\n");
    return;
  }

  bool all_done = false;
  while(!all_done) {
    struct AQLAndPacket packet = {};
    memset(&packet, 0, sizeof(AQLAndPacket));
    packet.header = AQL_PACKET_INVALID;
    int i;
    for (i = 0; i < AQL_MAX_SIGNAL_COUNT; i++) {
      packet.dep_signals[i] = ((chunk_info_t*)(dep_event->event->data))->start_address;
      POCL_MSG_PRINT_INFO("Creating AND barrier depending on signal id=%" PRIu64 " at address %" PRIu64 " \n", dep_event->event->id, packet.dep_signals[i]);
      dep_event = dep_event->next;
      if (dep_event == NULL) {
        all_done = true;
        break;
      }
    }
    packet.signal_count = i + 1;

    POCL_LOCK(D->AQLQueueLock);
    uint32_t queue_length = D->Dev->CQMemory->Size / AQL_PACKET_LENGTH - 1;

    uint32_t write_iter = D->Dev->CQMemory->Read32(ACCEL_CQ_WRITE);
    uint32_t read_iter = D->Dev->CQMemory->Read32(ACCEL_CQ_READ);
    while (write_iter >= read_iter + queue_length) {
      //POCL_MSG_PRINT_INFO("write_iter=%u, read_iter=%u length=%u", write_iter, read_iter, queue_length);
      read_iter = D->Dev->CQMemory->Read32(ACCEL_CQ_READ);
    }
    uint32_t packet_loc = (write_iter & (queue_length - 1)) * AQL_PACKET_LENGTH + AQL_PACKET_LENGTH;
    D->Dev->CQMemory->CopyToMMAP(packet_loc + D->Dev->CQMemory->PhysAddress,
                                   &packet, 64);

    D->Dev->CQMemory->Write16(packet_loc,
                          (1<<AQL_PACKET_BARRIER_AND) | AQL_PACKET_BARRIER);

    POCL_MSG_PRINT_INFO(
      "accel: Handed off and barrier, write iter=%u\n", write_iter);
    // Increment queue index
    D->Dev->CQMemory->Write32(ACCEL_CQ_WRITE, write_iter + 1);

    POCL_UNLOCK(D->AQLQueueLock);
  }
}

void pocl_accel_run(void *data, _cl_command_node *cmd) {}

void submit_kernel_packet(AccelData *D, _cl_command_node *cmd) {
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  // struct pocl_context *pc = &cmd->command.run.pc;

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0 || pc->num_groups[2] == 0)
    return;

  // First pass to figure out total argument size
  size_t arg_size = 0;
  for (i = 0; i < meta->num_args; ++i) {
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      arg_size += sizeof(size_t);
    } else {
      arg_size += meta->arg_info[i].type_size;
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
      POCL_ABORT_UNIMPLEMENTED("accel: local arguments");
    else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      /* It's legal to pass a NULL pointer to clSetKernelArguments. In
         that case we must pass the same NULL forward to the kernel.
         Otherwise, the user must have created a buffer with per device
         pointers stored in the cl_mem. */
      if (al->value == NULL) {
        *(size_t *)current_arg = 0;
      } else {
        // accel doesn't support SVM pointers
        assert(al->is_svm == 0);
        cl_mem m = (*(cl_mem *)(al->value));
        auto chunk =
            (chunk_info_t *)m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
        size_t buffer = (size_t)chunk->start_address;
        buffer += al->offset;
        if (D->Dev->RelativeAddressing) {
          buffer -= D->Dev->DataMemory->PhysAddress;
        }
        *(size_t *)current_arg = buffer;
      }
      current_arg += D->Dev->PointerSize;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      POCL_ABORT_UNIMPLEMENTED("accel: image arguments");
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_ABORT_UNIMPLEMENTED("accel: sampler arguments");
    } else {
      size_t size = meta->arg_info[i].type_size;
      memcpy(current_arg, &al->value, size);
      current_arg += size;
    }
  }

  scheduleNDRange(D, cmd, arg_size, arguments);
  free(arguments);
}

void pocl_accel_free_event_data (cl_event event)
{
  if(event->data != NULL){
    pocl_free_chunk((chunk_info_t*)event->data);
    event->data = NULL;
  }
}

void* runningThreadFunc(void*)
{
  while (!runningJoinRequested)
  {
    POCL_LOCK(runningLock);
    if (runningList) {
      _cl_command_node *Node = NULL;
      _cl_command_node *tmp = NULL;
      DL_FOREACH_SAFE(runningList, Node, tmp)
      {
        if (isEventDone((AccelData*)Node->device->data, Node->event)) {
          DL_DELETE(runningList, Node);
	  POCL_UNLOCK(runningLock);
	  POCL_UPDATE_EVENT_COMPLETE_MSG (Node->event, "Accel, asynchronous NDRange    ");
          POCL_LOCK(runningLock);
	}
      }
    }
    POCL_UNLOCK(runningLock);
    usleep(1000);
  }
  return NULL;
}
