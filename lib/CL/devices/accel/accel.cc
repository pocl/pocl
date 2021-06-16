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

#define ACCEL_INFO_IMEM_SIZE (0x314)
#define ACCEL_INFO_IMEM_START_LOW (0x318)
#define ACCEL_INFO_IMEM_START_HIGH (0x31C)

#define ACCEL_INFO_CQMEM_SIZE_LOW (0x320)
#define ACCEL_INFO_CQMEM_SIZE_HIGH (0x324)
#define ACCEL_INFO_CQMEM_START_LOW (0x328)
#define ACCEL_INFO_CQMEM_START_HIGH (0x32C)

#define ACCEL_INFO_DMEM_SIZE_LOW (0x330)
#define ACCEL_INFO_DMEM_SIZE_HIGH (0x334)
#define ACCEL_INFO_DMEM_START_LOW (0x338)
#define ACCEL_INFO_DMEM_START_HIGH (0x33C)

#define ACCEL_INFO_FEATURE_FLAGS_LOW (0x340)
#define ACCEL_INFO_FEATURE_FLAGS_HIGH (0x344)

#define ACCEL_FF_BIT_AXI_MASTER (1 << 0)
#define ACCEL_FF_BIT_W_IMEM_START (1 << 1)
#define ACCEL_FF_BIT_W_DMEM_START (1 << 2)
#define ACCEL_FF_BIT_PAUSE (1 << 3)

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
  POCL_CDBI_LEDBLINK = 3,
  POCL_CDBI_COUNTRED = 4,
};

struct AQLDispatchPacket {
  uint16_t header;
  uint16_t dimensions;

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
  uint64_t kernarg_address;

  uint64_t reserved;

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
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_LEDBLINK, "pocl.ledblink",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "input", PTR_ARG, GLOBAL_AS)}),
                        BIKD(POCL_CDBI_COUNTRED, "pocl.countred",
                             {BIArg("int*", "input", PTR_ARG, GLOBAL_AS),
                              BIArg("int*", "output", PTR_ARG, GLOBAL_AS)})};

class MMAPRegion {
public:
  MMAPRegion() : PhysAddress(0), Size(0), Data(nullptr) {}

  void Map(size_t Address, size_t RegionSize, int mem_fd) {
    PhysAddress = Address;
    Size = RegionSize;
    if (Size == 0) {
      return;
    }
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("accel: mmap'ing from address 0x%zx with size %zu\n",
                        Address, RegionSize);
#endif
    long page_size = sysconf(_SC_PAGESIZE);
    size_t roundDownAddress = (Address / page_size) * page_size;
    size_t difference = Address - roundDownAddress;
    Data = mmap(0, Size + difference, PROT_READ | PROT_WRITE, MAP_SHARED,
                mem_fd, roundDownAddress);
    assert(Data != MAP_FAILED && "MMAPRegion mapping failed");
    Data = (void *)((char *)Data + difference);
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
  // Used in emulator to hack the MMAP to work with just virtually contiguous
  // memory
  void Set(void *Address, size_t RegionSize) {
    PhysAddress = (size_t)Address;
    Data = Address;
    Size = RegionSize;
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

  uint64_t Read64(size_t offset) {
#ifdef ACCEL_MMAP_DEBUG
    POCL_MSG_PRINT_INFO("MMAP: Reading from physical address 0x%zx with "
                        "offset 0x%zx\n",
                        PhysAddress, offset);
#endif
    assert(Data && "No pointer to MMAP'd region; read before mapping?");
    assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
    auto value =
        static_cast<volatile uint64_t *>(Data)[offset / sizeof(uint64_t)];
    return value;
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

struct emulation_data_t {
  int Emulating;
  pthread_t emulate_thread;
  void *emulating_address;
  volatile int emulate_exit_called;
  volatile int emulate_init_done;
};

struct AccelData {
  size_t BaseAddress;

  MMAPRegion ControlMemory;
  MMAPRegion InstructionMemory;
  MMAPRegion DataMemory;
  MMAPRegion CQMemory;
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

  int RelativeAddressing;
  emulation_data_t EmulationData;
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
  d->DataMemory.CopyToMMAP(dst, src_host_ptr, size);
}

void pocl_accel_read(void *data, void *__restrict__ dst_host_ptr,
                     pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                     size_t offset, size_t size) {
  chunk_info_t *chunk = (chunk_info_t *)src_mem_id->mem_ptr;
  size_t src = chunk->start_address + offset;
  AccelData *d = (AccelData *)data;
  POCL_MSG_PRINT_INFO("accel: Copying 0x%zu bytes from 0x%zu\n", size, src);
  d->DataMemory.CopyFromMMAP(dst_host_ptr, src, size);
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
  if ((mem_obj->flags & CL_MEM_COPY_HOST_PTR) ||
      ((mem_obj->flags & CL_MEM_USE_HOST_PTR) && host_ptr != NULL)) {
    /* TODO:
       CL_MEM_USE_HOST_PTR must synch the buffer after execution
       back to the host's memory in case it's used as an output (?). */
    data->DataMemory.CopyToMMAP(chunk->start_address, host_ptr, mem_obj->size);
  }

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
  emulation_data_t *E = &(D->EmulationData);
  E->emulating_address = NULL;
  int mem_fd = -1;
  // Recognize whether we are emulating or not
  if (D->BaseAddress == EMULATING_ADDRESS) {
    E->Emulating = 1;
    // The principle of the emulator is that instead of mmapping a real
    // accelerator, we just allocate a regular array, which corresponds
    // to the mmap. The accel_emulate function is working in another thread
    // and will fill that array and respond to the driver asynchronously.
    // The driver doesn't really need to know about emulating except
    // in the initial mapping of the accelerator
    E->emulating_address = calloc(1, EMULATING_MAX_SIZE);
    assert(E->emulating_address != NULL && "Emulating calloc failed\n");

    D->ControlMemory.Set(E->emulating_address, 1024);

    // Create emulator thread
    E->emulate_exit_called = 0;
    E->emulate_init_done = 0;
    pthread_create(&(E->emulate_thread), NULL, emulate_accel, E);
    while (!E->emulate_init_done)
      ; // Wait for the thread to initialize
    POCL_MSG_PRINT_INFO("accel: started emulating\n");
  } else {
    E->Emulating = 0;
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd == -1) {
      POCL_ABORT("Could not open /dev/mem\n");
    }
    D->ControlMemory.Map(D->BaseAddress, ACCEL_DEFAULT_CTRL_SIZE, mem_fd);
  }

  if (D->ControlMemory.Read32(ACCEL_INFO_CORE_COUNT) != 1) {
    POCL_ABORT_UNIMPLEMENTED("Multicore accelerators");
  }

  uint64_t feature_flags =
      D->ControlMemory.Read64(ACCEL_INFO_FEATURE_FLAGS_LOW);

  // Turn on the relative addressing if the target has no axi master.
  D->RelativeAddressing = (feature_flags & ACCEL_FF_BIT_AXI_MASTER) ? (0) : (1);
  // Reset accelerator
  D->ControlMemory.Write32(ACCEL_AQL_WRITE_LOW,
                           -D->ControlMemory.Read32(ACCEL_AQL_WRITE_LOW));
  D->ControlMemory.Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_RESET_CMD);

  uint32_t ctrl_size = D->ControlMemory.Read32(ACCEL_INFO_CTRL_SIZE);
  uint32_t imem_size = D->ControlMemory.Read32(ACCEL_INFO_IMEM_SIZE);
  uint32_t cq_size = D->ControlMemory.Read32(ACCEL_INFO_CQMEM_SIZE_LOW);
  uint32_t dmem_size = D->ControlMemory.Read32(ACCEL_INFO_DMEM_SIZE_LOW);

  uintptr_t imem_start = D->ControlMemory.Read64(ACCEL_INFO_IMEM_START_LOW);
  uintptr_t cq_start = D->ControlMemory.Read64(ACCEL_INFO_CQMEM_START_LOW);
  uintptr_t dmem_start = D->ControlMemory.Read64(ACCEL_INFO_DMEM_START_LOW);

  if (D->RelativeAddressing) {
    POCL_MSG_PRINT_INFO("Accel: Enabled relative addressing\n");
    cq_start += D->ControlMemory.PhysAddress;
    imem_start += D->ControlMemory.PhysAddress;
    dmem_start += D->ControlMemory.PhysAddress;
  }

  if (E->Emulating) {
    // If emulating, skip the mmaping and just directly set the MMAPRegion's
    // values
    D->InstructionMemory.Set((char *)imem_start, imem_size);
    D->DataMemory.Set((char *)dmem_start, dmem_size);
    D->CQMemory.Set((char *)cq_start, cq_size);
  } else {
    // Does the proper mmaping based on the physical address
    D->InstructionMemory.Map(imem_start, imem_size, mem_fd);
    D->CQMemory.Map(cq_start, cq_size, mem_fd);
    D->DataMemory.Map(dmem_start, dmem_size, mem_fd);
  }
  pocl_init_mem_region(&D->AllocRegion, dmem_start, dmem_size);

  // memory mapping done
  close(mem_fd);

  POCL_MSG_PRINT_INFO("accel: mmap done\n");
  // Initialize AQL queue by setting all headers to invalid
  for (uint32_t i = 0; i < cq_size; i += AQL_PACKET_LENGTH) {
    D->CQMemory.Write16(i, AQL_PACKET_INVALID);
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

  if (E->Emulating) {
    std::cout << "Custom emulation device " << j << " initialized" << std::endl;
  } else {
    std::cout << "Custom device " << j << " initialized" << std::endl;
  }
  return CL_SUCCESS;
}

cl_int pocl_accel_uninit(unsigned /*j*/, cl_device_id device) {
  POCL_MSG_PRINT_INFO("accel: uninit\n");
  AccelData *D = (AccelData *)device->data;
  emulation_data_t *E = &(D->EmulationData);
  if (E->Emulating) {
    POCL_MSG_PRINT_INFO("accel: freeing emulated accel");
    E->emulate_exit_called = 1; // signal for the emulator to stop
    pthread_join(E->emulate_thread, NULL);
    free((void *)E->emulating_address); // from
                                        // calloc(emulating_address)
  } else {
    D->ControlMemory.Unmap();
    D->InstructionMemory.Unmap();
    D->DataMemory.Unmap();
    D->CQMemory.Unmap();
  }
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

chunk_info_t* scheduleNDRange(AccelData *data, _cl_command_run *run, size_t arg_size,
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
  data->DataMemory.Write32(signalAddress - data->DataMemory.PhysAddress, 0);
  // Set arguments
  data->DataMemory.CopyToMMAP(argsAddress, arguments, arg_size);

  struct AQLDispatchPacket packet = {};

  packet.header = AQL_PACKET_INVALID;
  packet.dimensions = run->pc.work_dim; // number of dimensions

  packet.workgroup_size_x = run->pc.local_size[0];
  packet.workgroup_size_y = run->pc.local_size[1];
  packet.workgroup_size_z = run->pc.local_size[2];

  packet.grid_size_x = run->pc.local_size[0] * run->pc.num_groups[0];
  packet.grid_size_y = run->pc.local_size[1] * run->pc.num_groups[1];
  packet.grid_size_z = run->pc.local_size[2] * run->pc.num_groups[2];

  packet.kernel_object = kernelID;

  if (data->RelativeAddressing) {
    packet.kernarg_address = argsAddress - data->DataMemory.PhysAddress;
    packet.completion_signal = signalAddress - data->DataMemory.PhysAddress;
  } else {
    packet.kernarg_address = argsAddress;
    packet.completion_signal = signalAddress;
  }

  POCL_LOCK(data->AQLQueueLock);

  uint32_t queue_length = data->CQMemory.Size / AQL_PACKET_LENGTH;
  uint32_t write_iter = data->ControlMemory.Read32(ACCEL_AQL_WRITE_LOW);
  uint32_t read_iter = data->ControlMemory.Read32(ACCEL_AQL_READ_LOW);
  while (write_iter >= read_iter + queue_length) {
    read_iter = data->ControlMemory.Read32(ACCEL_AQL_READ_LOW);
  }
  uint32_t packet_loc = (write_iter & (queue_length - 1)) * AQL_PACKET_LENGTH;
  data->CQMemory.CopyToMMAP(packet_loc + data->CQMemory.PhysAddress, &packet,
                            64);
  // finally, set header as not-invalid
  data->CQMemory.Write16(packet_loc,
                         AQL_PACKET_KERNEL_DISPATCH | AQL_PACKET_BARRIER);

  POCL_MSG_PRINT_INFO(
      "accel: Handed off a packet for execution, write iter=%u\n", write_iter);
  // Increment queue index
  if (data->EmulationData.Emulating) {
    // The difference between emulation and actual accelerators is that the
    // actual accelerator automagically increments the write_iter with the
    // value of the second parameter.
    data->ControlMemory.Write32(ACCEL_AQL_WRITE_LOW, write_iter + 1);
  } else {
    data->ControlMemory.Write32(ACCEL_AQL_WRITE_LOW, 1);
  }

  POCL_UNLOCK(data->AQLQueueLock);

  return chunk;
}

int waitOnEvent(AccelData *data, size_t event) {
  size_t offset = event - data->DataMemory.PhysAddress;
  uint32_t status;
  do {
    usleep(20);
    status = data->DataMemory.Read32(offset);
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
        if (D->RelativeAddressing) {
          buffer -= D->DataMemory.PhysAddress;
        }
        *(size_t *)current_arg = buffer;
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

  chunk_info_t* chunk = scheduleNDRange(D, &cmd->command.run, arg_size, arguments);
  size_t event = chunk->start_address;
  free(arguments);
  int fail = waitOnEvent(D, event);
  free_chunk(chunk);
  if (fail) {
    POCL_MSG_ERR("accel: command execution returned failure with kernel %s\n",
                 kernel->name);
  } else {
    POCL_MSG_PRINT_INFO("accel: successfully executed kernel %s\n",
                        kernel->name);
  }
}

/*
 * AlmaIF v1 based accel emulator
 * Base_address is a preallocated emulation array that corresponds to the
 * memory map of the accelerator
 * Does operations 0,1,2,3,4
 * */
void *emulate_accel(void *E_void) {

  emulation_data_t *E = (emulation_data_t *)E_void;
  void *base_address = E->emulating_address;

  uint32_t ctrl_size = 1024;
  uint32_t imem_size = 0;
  uint32_t dmem_size = 2097152;
  // The accelerator can choose the size of the queue (must be a power-of-two)
  // Can be even 1, to make the packet handling easiest with static offsets
  uint32_t queue_length = 2;
  uint32_t cqmem_size = queue_length * AQL_PACKET_LENGTH;

  // The accelerator can set the starting addresses
  // Even the order can be changed if the accelerator wants to
  // Here packing the memory regions tighly as an example.
  uintptr_t imem_start = (uintptr_t)base_address + ctrl_size;
  uintptr_t cqmem_start = imem_start + imem_size;
  uintptr_t dmem_start = cqmem_start + cqmem_size;

  volatile uint32_t *Control = (uint32_t *)base_address;
  volatile uint8_t *Instruction = (uint8_t *)imem_start;
  volatile uint8_t *CQ = (uint8_t *)cqmem_start;
  volatile uint8_t *Data = (uint8_t *)dmem_start;

  // Set initial values for info registers:
  Control[ACCEL_INFO_DEV_CLASS / 4] = 0xE; // Unused
  Control[ACCEL_INFO_DEV_ID / 4] = 0;      // Unused
  Control[ACCEL_INFO_IF_TYPE / 4] = 1;
  Control[ACCEL_INFO_CORE_COUNT / 4] = 1;
  Control[ACCEL_INFO_CTRL_SIZE / 4] = 1024;

  // The emulation doesn't use Instruction/Configuration memory. This memory
  // space is a place to write accelerator specific configuration values
  // that are written BEFORE hw reset is deasserted.
  // E.g. program binaries of a processor-based accelerator
  Control[ACCEL_INFO_IMEM_SIZE / 4] = 0;
  Control[ACCEL_INFO_IMEM_START_LOW / 4] = (uint32_t)imem_start;
  Control[ACCEL_INFO_IMEM_START_HIGH / 4] = (uint32_t)(imem_start >> 32);

  Control[ACCEL_INFO_CQMEM_SIZE_LOW / 4] = cqmem_size;
  Control[ACCEL_INFO_CQMEM_START_LOW / 4] = (uint32_t)cqmem_start;
  Control[ACCEL_INFO_CQMEM_START_HIGH / 4] = (uint32_t)(cqmem_start >> 32);

  Control[ACCEL_INFO_DMEM_SIZE_LOW / 4] = dmem_size;
  Control[ACCEL_INFO_DMEM_START_LOW / 4] = (uint32_t)dmem_start;
  Control[ACCEL_INFO_DMEM_START_HIGH / 4] = (uint32_t)(dmem_start >> 32);

  uint32_t feature_flags_low = ACCEL_FF_BIT_AXI_MASTER;
  Control[ACCEL_INFO_FEATURE_FLAGS_LOW / 4] = feature_flags_low;

  // Signal the driver that the initial values are set
  // (in hardware this signal is probably not needed, since the values are
  // initialized in hw reset)
  E->emulate_init_done = 1;
  POCL_MSG_PRINT_INFO("accel emulate: Emulator initialized");

  int read_iter = 0;
  Control[ACCEL_AQL_READ_LOW / 4] = read_iter;
  // Accelerator is in infinite loop to process the commands
  // For emulating purposes we include the exit signal that the driver can
  // use to terminate the emulating thread. In hw this could be
  // a while(1) loop.
  while (!E->emulate_exit_called) {

    // Don't start computing anything before soft reset is lifted.
    // (This could probably be outside of the loop)
    int reset = Control[ACCEL_CONTROL_REG_COMMAND / 4];
    if (reset != ACCEL_CONTINUE_CMD) {
      continue;
    }

    // Compute packet location
    uint32_t packet_loc = (read_iter & (queue_length - 1)) * AQL_PACKET_LENGTH;
    struct AQLDispatchPacket *packet =
        (struct AQLDispatchPacket *)(CQ + packet_loc);

    // The driver will mark the packet as not INVALID when it wants us to
    // compute it
    if (packet->header == AQL_PACKET_INVALID) {
      continue;
    }

    POCL_MSG_PRINT_INFO("accel emulate: Found valid AQL_packet from location "
                        "%u, starting parsing:",
                        packet_loc);
    POCL_MSG_PRINT_INFO(
        "accel emulate: kernargs are at %lu\n",
        packet->kernarg_address);
    // Find the 3 pointers
    // Pointer size can be different on different systems
    // Also the endianness might need some attention in the real case.
#define PTR_SIZE sizeof(uint32_t *)
    union args_u {
      uint32_t *ptrs[3];
      uint8_t values[3 * PTR_SIZE];
    } args;
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < PTR_SIZE; k++) {
       args.values[PTR_SIZE * i + k] =
            *(uint8_t*)(packet->kernarg_address + PTR_SIZE * i + k);
      }
    }
    uint32_t *arg0 = args.ptrs[0];
    uint32_t *arg1 = args.ptrs[1];
    uint32_t *arg2 = args.ptrs[2];

    POCL_MSG_PRINT_INFO("accel emulate: FOUND args arg0=%p, arg1=%p, arg2=%p\n",
                        arg0, arg1, arg2);

    // Check how many dimensions are in use, and set the unused ones to 1.
    int dim_x = packet->grid_size_x;
    int dim_y = (packet->dimensions >= 2) ? (packet->grid_size_y) : 1;
    int dim_z = (packet->dimensions == 3) ? (packet->grid_size_z) : 1;

    int red_count = 0;
    POCL_MSG_PRINT_INFO(
        "accel emulate: Parsing done: starting loops with dims (%i,%i,%i)",
        dim_x, dim_y, dim_z);
    for (int x = 0; x < dim_x; x++) {
      for (int y = 0; y < dim_y; y++) {
        for (int z = 0; z < dim_z; z++) {
          // Linearize grid
          int idx = x * dim_y * dim_z + y * dim_z + z;
          // Do the operation based on the kernel_object (integer id)
          switch (packet->kernel_object) {
          case (POCL_CDBI_COPY):
            arg1[idx] = arg0[idx];
            break;
          case (POCL_CDBI_ADD32):
            arg2[idx] = arg0[idx] + arg1[idx];
            break;
          case (POCL_CDBI_MUL32):
            arg2[idx] = arg0[idx] * arg1[idx];
            break;
          case (POCL_CDBI_COUNTRED):
            uint32_t pixel = arg0[idx];
            uint8_t pixel_r = pixel & 0xFF;
            if (pixel_r > 100) {
              red_count++;
            }
          }
        }
      }
    }
    if (packet->kernel_object == POCL_CDBI_LEDBLINK) {
      std::cout << "Emulation blinking " << dim_x << " led(s) at interval "
                << arg0[0] << " us " << arg1[0] << " times" << std::endl;
    }
    if (packet->kernel_object == POCL_CDBI_COUNTRED) {
      arg1[0] = red_count;
    }

    POCL_MSG_PRINT_INFO("accel emulate: Kernel done");

    //Completion signal is given as absolute address
    *(uint32_t*) packet->completion_signal = 1;
    packet->header = AQL_PACKET_INVALID;

    read_iter++; // move on to the next AQL packet
    Control[ACCEL_AQL_READ_LOW / 4] = read_iter;
  }
}
