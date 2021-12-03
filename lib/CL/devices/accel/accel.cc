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

#include "accel.h"
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

// MMAPRegion debug prints get quite spammy
// #define ACCEL_MMAP_DEBUG

#define ACCEL_DEFAULT_CTRL_SIZE (1024)

#define ACCEL_STATUS_REG (0x00)
#define ACCEL_STATUS_REG_PC (0x04)
#define ACCEL_STATUS_REG_CC_LOW (0x08)
#define ACCEL_STATUS_REG_CC_HIGH (0x0C)
#define ACCEL_STATUS_REG_SC_LOW (0x10)
#define ACCEL_STATUS_REG_SC_HIGH (0x14)

#define ACCEL_AQL_READ_LOW (0x100)
#define ACCEL_AQL_READ_HIGH (0x104)
#define ACCEL_AQL_WRITE_LOW (0x108)
#define ACCEL_AQL_WRITE_HIGH (0x10C)

#define ACCEL_CONTROL_REG_COMMAND (0x200)

#define ACCEL_RESET_CMD (1)
#define ACCEL_CONTINUE_CMD (2)
#define ACCEL_BREAK_CMD (4)

#define ACCEL_INFO_DEV_CLASS (0x300)
#define ACCEL_INFO_DEV_ID (0x304)
#define ACCEL_INFO_IF_TYPE (0x308)
#define ACCEL_INFO_CORE_COUNT (0x30C)
#define ACCEL_INFO_CTRL_SIZE (0x310)
#define ACCEL_INFO_DMEM_SIZE (0x314)
#define ACCEL_INFO_IMEM_SIZE (0x318)
#define ACCEL_INFO_PMEM_SIZE (0x31C)

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)

#define AQL_PACKET_BARRIER (1 << 8)

#define AQL_PACKET_LENGTH (64)

enum BuiltinKernelId : uint16_t {
  // CD = custom device, BI = built-in
  // 1D array byte copy, get_global_size(0) defines the size of data to copy
  // kernel prototype: pocl.copy(char *input, char *output)
  POCL_CDBI_COPY = 0,
  POCL_CDBI_ADD32 = 1,
  POCL_CDBI_MUL32 = 2,
};

struct AQLDispatchPacket {
  uint16_t header;
  uint16_t setup;

  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;

  uint16_t reserved0;

  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;

  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  uint32_t kernarg_address;

  uint32_t reserved1;
  uint64_t reserved2;

  uint64_t completion_signal;
};

// An initialization wrapper for kernel argument metadatas.
struct BIArg : public pocl_argument_info {
  BIArg(const char *TypeName, const char *Name, pocl_argument_type Type,
        cl_kernel_arg_address_qualifier AQ) {
    type = Type;
    address_qualifier = AQ;
    type_name = strdup(TypeName);
    name = strdup(Name);
  }

  ~BIArg() {
    free(name);
    free(type_name);
  }
};

// An initialization wrapper for kernel metadatas.
// BIKD = Built-in Kernel Descriptor
struct BIKD : public pocl_kernel_metadata_t {
  BIKD(BuiltinKernelId KernelId, const char *KernelName,
       const std::vector<pocl_argument_info> &ArgInfos);

  ~BIKD() {
    delete[] arg_info;
    free(name);
  }

  BuiltinKernelId KernelId;
};

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
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)})};

class MMAPRegion {
public:
  MMAPRegion() : PhysAddress(0), Size(0), Data(nullptr) {}

  void Map(size_t Address, size_t RegionSize, int mem_fd) {
    PhysAddress = Address;
    Size = RegionSize;
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("accel: mmap'ing from address 0x%zx with size %zu\n",
                        Address, RegionSize);
#endif
    Data = mmap(0, Size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, Address);
    assert(Data != MAP_FAILED && "MMAPRegion mapping failed");
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("accel: got address %p\n", Data);
#endif
  }

  void Unmap() {
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("accel: munmap'ing from address 0x%zx\n", PhysAddress);
#endif
    if (Data) {
      munmap(Data, Size);
      Data = NULL;
    }
  }

  uint32_t Read32(size_t offset) {
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("MMAP: Reading from physical address 0x%zx with "
                        "offset 0x%zx\n",
                        PhysAddress, offset);
#endif
    assert(Data && "No pointer to MMAP'd region; read before mapping?");
    assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
    auto value =
        static_cast<volatile uint32_t *>(Data)[offset / sizeof(uint32_t)];
    return value;
  }

  void Write32(size_t offset, uint32_t value) {
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("MMAP: Writing to physical address 0x%zx with "
                        "offset 0x%zx\n",
                        PhysAddress, offset);
#endif
    assert(Data && "No pointer to MMAP'd region; write before mapping?");
    assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
    static_cast<volatile uint32_t *>(Data)[offset / sizeof(uint32_t)] = value;
  }

  void Write16(size_t offset, uint16_t value) {
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("MMAP: Writing to physical address 0x%zx with "
                        "offset 0x%zx\n",
                        PhysAddress, offset);
#endif
    assert(Data && "No pointer to MMAP'd region; write before mapping?");
    assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
    static_cast<volatile uint16_t *>(Data)[offset / sizeof(uint16_t)] = value;
  }

  size_t VirtualToPhysical(void *ptr) {
    size_t offset = ((size_t)ptr) - (size_t)Data;
    assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
    return offset + PhysAddress;
  }

  void CopyToMMAP(size_t destination, const void *source, size_t bytes) {
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("MMAP: Writing 0x%zx bytes to buffer at 0x%zx with "
                        "address 0x%zx\n",
                        bytes, PhysAddress, destination);
#endif
    auto src = (char *)source;
    size_t offset = destination - PhysAddress;
    assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
    auto dst = offset + static_cast<volatile char *>(Data);
    for (size_t i = 0; i < bytes; ++i) {
      dst[i] = src[i];
    }
  }

  void CopyFromMMAP(void *destination, size_t source, size_t bytes) {
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("MMAP: Reading 0x%zx bytes from buffer at 0x%zx "
                        "with address 0x%zx\n",
                        bytes, PhysAddress, source);
#endif
    auto dst = (char *)destination;
    size_t offset = source - PhysAddress;
    assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
    auto src = offset + static_cast<volatile char *>(Data);
    for (size_t i = 0; i < bytes; ++i) {
      dst[i] = src[i];
    }
  }

  size_t PhysAddress;
  size_t Size;

private:
  void *Data;
};

struct AccelData {
  size_t BaseAddress;

  MMAPRegion ControlMemory;
  MMAPRegion InstructionMemory;
  MMAPRegion DataMemory;
  MMAPRegion ParameterMemory;
  memory_region_t AllocRegion;

  std::set<BIKD *> SupportedKernels;
  // List of commands ready to be executed.
  _cl_command_node *ReadyList;
  // List of commands not yet ready to be executed.
  _cl_command_node *CommandList;
  // Lock for command list related operations.
  pocl_lock_t CommandListLock;

  // Lock for device-side command queue manipulation
  pocl_lock_t AQLQueueLock;
};

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
  ops->flush = ops->join = pocl_accel_join;

  ops->write = pocl_accel_write;
  ops->read = pocl_accel_read;

  ops->notify = pocl_accel_notify;

  ops->broadcast = pocl_broadcast;
  ops->run = pocl_accel_run;

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
  d->ParameterMemory.CopyToMMAP(dst, src_host_ptr, size);
}

void pocl_accel_read(void *data, void *__restrict__ dst_host_ptr,
                     pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                     size_t offset, size_t size) {
  chunk_info_t *chunk = (chunk_info_t *)src_mem_id->mem_ptr;
  size_t src = chunk->start_address + offset;
  AccelData *d = (AccelData *)data;
  POCL_MSG_PRINT_INFO("accel: Copying 0x%zu bytes from 0x%zu\n", size, src);
  d->ParameterMemory.CopyFromMMAP(dst_host_ptr, src, size);
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

  chunk = pocl_alloc_buffer_from_region(&data->AllocRegion, mem_obj->size);
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
    if (!found) {
      POCL_ABORT("accel: Unknown Kernel ID (%lu) given\n", token);
    }
  }

  dev->builtin_kernel_list = strdup(supportedList.c_str());

  POCL_MSG_PRINT_INFO("accel: accelerator at 0x%zx with %zu builtin kernels\n",
                      D->BaseAddress, D->SupportedKernels.size());

  int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (mem_fd == -1) {
    POCL_ABORT("Could not open /dev/mem\n");
  }

  D->ControlMemory.Map(D->BaseAddress, ACCEL_DEFAULT_CTRL_SIZE, mem_fd);

  if (D->ControlMemory.Read32(ACCEL_INFO_CORE_COUNT) != 1) {
    POCL_ABORT_UNIMPLEMENTED("Multicore accelerators");
  }

  // Reset accelerator
  D->ControlMemory.Write32(ACCEL_AQL_WRITE_LOW,
                           -D->ControlMemory.Read32(ACCEL_AQL_WRITE_LOW));
  D->ControlMemory.Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_RESET_CMD);

  uint32_t ctrl_size = D->ControlMemory.Read32(ACCEL_INFO_CTRL_SIZE);
  uint32_t imem_size = D->ControlMemory.Read32(ACCEL_INFO_IMEM_SIZE);
  uint32_t dmem_size = D->ControlMemory.Read32(ACCEL_INFO_DMEM_SIZE);
  uint32_t pmem_size = D->ControlMemory.Read32(ACCEL_INFO_PMEM_SIZE);

  uint32_t max_region =
      std::max(std::max(ctrl_size, imem_size), std::max(dmem_size, pmem_size));

  D->InstructionMemory.Map(D->BaseAddress + max_region, imem_size, mem_fd);
  D->DataMemory.Map(D->BaseAddress + 2 * max_region, dmem_size, mem_fd);
  D->ParameterMemory.Map(D->BaseAddress + 3 * max_region, pmem_size, mem_fd);

  init_mem_region(&D->AllocRegion, D->ParameterMemory.PhysAddress, pmem_size);

  // memory mapping done
  close(mem_fd);

  // Initialize AQL queue by setting all headers to invalid
  for (uint32_t i = 0; i < dmem_size; i += AQL_PACKET_LENGTH) {
    D->DataMemory.Write16(i, AQL_PACKET_INVALID);
  }

  // Lift accelerator reset
  D->ControlMemory.Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_CONTINUE_CMD);

  // This would be more logical as a per builtin kernel value?
  // Either way, the minimum is 3 for a device
  dev->max_work_item_dimensions = 3;
  dev->max_work_group_size = dev->max_work_item_sizes[0] =
      dev->max_work_item_sizes[1] = dev->max_work_item_sizes[2] = INT_MAX;

  D->ReadyList = NULL;
  D->CommandList = NULL;
  POCL_INIT_LOCK(D->CommandListLock);
  POCL_INIT_LOCK(D->AQLQueueLock);

  std::cout << "Custom device " << j << " initialized" << std::endl;
  return CL_SUCCESS;
}

cl_int pocl_accel_uninit(unsigned /*j*/, cl_device_id device) {
  POCL_MSG_PRINT_INFO("accel: uninit\n");
  AccelData *D = (AccelData *)device->data;
  D->ControlMemory.Unmap();
  D->InstructionMemory.Unmap();
  D->DataMemory.Unmap();
  D->ParameterMemory.Unmap();
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
    CDL_DELETE(D.ReadyList, Node);
    POCL_UNLOCK(D.CommandListLock);
    pocl_exec_command(Node);
    POCL_LOCK(D.CommandListLock);
  }
  return;
}

void pocl_accel_submit(_cl_command_node *Node, cl_command_queue /*CQ*/) {

  Node->ready = 1;

  struct AccelData *D = (AccelData *)Node->device->data;
  POCL_LOCK(D->CommandListLock);
  pocl_command_push(Node, &D->ReadyList, &D->CommandList);

  POCL_UNLOCK_OBJ(Node->event);
  scheduleCommands(*D);
  POCL_UNLOCK(D->CommandListLock);
  return;
}

void pocl_accel_join(cl_device_id Device, cl_command_queue /*CQ*/) {

  struct AccelData *D = (AccelData *)Device->data;
  POCL_LOCK(D->CommandListLock);
  scheduleCommands(*D);
  POCL_UNLOCK(D->CommandListLock);
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

  if (pocl_command_is_ready(Event)) {
    if (Event->status == CL_QUEUED) {
      pocl_update_event_submitted(Event);
      POCL_LOCK(D.CommandListLock);
      CDL_DELETE(D.CommandList, Node);
      CDL_PREPEND(D.ReadyList, Node);
      scheduleCommands(D);
      POCL_UNLOCK(D.CommandListLock);
    }
    return;
  }
}

size_t scheduleNDRange(AccelData *data, _cl_command_run *run, size_t arg_size,
                       void *arguments) {
  int32_t kernelID = -1;
  for (auto supportedKernel : data->SupportedKernels) {
    if (strcmp(supportedKernel->name, run->kernel->name) == 0)
      kernelID = (int32_t)supportedKernel->KernelId;
  }

  if (kernelID == -1) {
    POCL_ABORT("accel: scheduled an NDRange with unsupported kernel\n");
  }
  // Additional space for a signal
  size_t extraAlloc = sizeof(uint32_t);
  chunk_info_t *chunk =
      pocl_alloc_buffer_from_region(&data->AllocRegion, arg_size + extraAlloc);
  assert(chunk && "Failed to allocate signal/argument buffer");

  POCL_MSG_PRINT_INFO("accel: allocated 0x%zx bytes for signal/arguments "
                      "from 0x%zx\n",
                      arg_size + extraAlloc, chunk->start_address);
  size_t signalAddress = chunk->start_address;
  size_t argsAddress = signalAddress + extraAlloc;
  // Set initial signal value
  data->ParameterMemory.Write32(
      signalAddress - data->ParameterMemory.PhysAddress, 0);
  // Set arguments
  data->ParameterMemory.CopyToMMAP(argsAddress, arguments, arg_size);

  struct AQLDispatchPacket packet = {};

  packet.header = AQL_PACKET_INVALID;
  packet.setup = run->pc.work_dim; // number of dimensions

  packet.workgroup_size_x = run->pc.local_size[0];
  packet.workgroup_size_y = run->pc.local_size[1];
  packet.workgroup_size_z = run->pc.local_size[2];

  packet.grid_size_x = run->pc.local_size[0] * run->pc.num_groups[0];
  packet.grid_size_y = run->pc.local_size[1] * run->pc.num_groups[1];
  packet.grid_size_z = run->pc.local_size[2] * run->pc.num_groups[2];

  packet.kernel_object = kernelID;
  packet.kernarg_address = argsAddress;
  packet.completion_signal = signalAddress;

  POCL_LOCK(data->AQLQueueLock);

  uint32_t queue_length = data->DataMemory.Size / AQL_PACKET_LENGTH;
  uint32_t write_iter = data->ControlMemory.Read32(ACCEL_AQL_WRITE_LOW);
  uint32_t read_iter = data->ControlMemory.Read32(ACCEL_AQL_READ_LOW);
  while (write_iter >= read_iter + queue_length) {
    read_iter = data->ControlMemory.Read32(ACCEL_AQL_READ_LOW);
  }
  uint32_t packet_loc = (write_iter & (queue_length - 1)) * AQL_PACKET_LENGTH;
  data->DataMemory.CopyToMMAP(packet_loc + data->DataMemory.PhysAddress,
                              &packet, 64);
  // finally, set header as not-invalid
  data->DataMemory.Write16(packet_loc,
                           AQL_PACKET_KERNEL_DISPATCH | AQL_PACKET_BARRIER);

  // Increment queue index
  data->ControlMemory.Write32(ACCEL_AQL_WRITE_LOW, 1);

  POCL_UNLOCK(data->AQLQueueLock);

  return chunk->start_address;
}

int waitOnEvent(AccelData *data, size_t event) {
  size_t offset = event - data->ParameterMemory.PhysAddress;
  uint32_t status;
  do {
    usleep(20000);
    status = data->ParameterMemory.Read32(offset);
  } while (status == 0);
  return status - 1;
}

void pocl_accel_run(void *data, _cl_command_node *cmd) {
  struct AccelData *D = (AccelData *)data;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;

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
        size_t phys = buffer + al->offset;
        *(size_t *)current_arg = phys;
      }
      current_arg += sizeof(size_t);
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

  size_t event = scheduleNDRange(D, &cmd->command.run, arg_size, arguments);
  free(arguments);
  int fail = waitOnEvent(D, event);
  if (fail) {
    POCL_MSG_ERR("accel: command execution returned failure with kernel %s\n",
                 kernel->name);
  } else {
    POCL_MSG_PRINT_INFO("accel: successfully executed kernel %s\n",
                        kernel->name);
  }
}
