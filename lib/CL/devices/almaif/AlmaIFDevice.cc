/* AlmaIFDevice.cc - Interface class for accessing AlmaIF device.
 * Responsible for setting up AlmaIF regions and device-backend specific
 * init (e.g. initializing a simulator or mmapping physical accelerator.)

   Copyright (c) 2022 Topi Leppänen / Tampere University

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

#include "AlmaIFDevice.hh"

#include "AlmaifShared.hh"
#include "AlmaifCompile.hh"

#include "bufalloc.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_timing.h"

#include <iostream>

AlmaIFDevice::AlmaIFDevice() {}

AlmaIFDevice::~AlmaIFDevice() {
  delete ControlMemory;
  delete InstructionMemory;
  delete CQMemory;
  delete DataMemory;
  delete ExternalMemory;
  memory_region_t *el, *tmp;
  LL_FOREACH_SAFE(AllocRegions, el, tmp) { free(el); }
}

void AlmaIFDevice::discoverDeviceParameters() {
  // Reset accelerator
  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_RESET_CMD);

  if (ControlMemory->Read32(ALMAIF_INFO_CORE_COUNT) != 1) {
    POCL_ABORT_UNIMPLEMENTED("Multicore accelerators");
  }

  uint32_t interface_version = ControlMemory->Read32(ALMAIF_INFO_IF_TYPE);

  if (interface_version == ALMAIF_VERSION_2) {
    /*Only AamuDSP should be using the old interface, if somethine else is,
     * you should modify this part to be more generic
    The mmap for opencl use case goes like this:
    BaseAddress                                       -->ControlMemory
    BaseAddress + segment_size                        --> Imem
    BaseAddress + 3*segment_size                      --> Dmem (for buffers)
    BaseAddress + 3*segment_size + Dmem_size - PRIVATE_MEM_SIZE - 4*64 --> Cqmem
    BaseAddress + 3*segment_size + Dmem_size - PRIVATE_MEM_SIZE        --> Local
    scratchpad memory for stack etc Where segment_size = 0x10000 (size of imem)
    */
    ImemSize = ControlMemory->Read32(ALMAIF_INFO_IMEM_SIZE_LEGACY);
    // CQSize = ControlMemory->Read32(ALMAIF_INFO_PMEM_SIZE_LEGACY);
    CQSize = 4 * 64;
    // DmemSize = ControlMemory->Read32(ALMAIF_INFO_PMEM_SIZE_LEGACY);
    int private_mem_size =
        pocl_get_int_option("POCL_ALMAIF_PRIVATE_MEM_SIZE", ALMAIF_DEFAULT_PRIVATE_MEM_SIZE);

    DmemSize = ControlMemory->Read32(ALMAIF_INFO_PMEM_SIZE_LEGACY) -
               private_mem_size - CQSize;
    PointerSize = 4;
    RelativeAddressing = false;

    uint32_t segment_size = ImemSize;
    ImemStart = segment_size;
    DmemStart = 3 * segment_size;
    CQStart = DmemStart + DmemSize;
    CQStart += ControlMemory->PhysAddress();
    ImemStart += ControlMemory->PhysAddress();
    DmemStart += ControlMemory->PhysAddress();
  } else if (interface_version == ALMAIF_VERSION_3) {
    uint64_t feature_flags =
        ControlMemory->Read64(ALMAIF_INFO_FEATURE_FLAGS_LOW);

    PointerSize = ControlMemory->Read32(ALMAIF_INFO_PTR_SIZE);
    // Turn on the relative addressing if the target has no axi master.
    RelativeAddressing =
        (feature_flags & ALMAIF_FF_BIT_AXI_MASTER) ? (false) : (true);

    ImemSize = ControlMemory->Read32(ALMAIF_INFO_IMEM_SIZE);
    CQSize = ControlMemory->Read32(ALMAIF_INFO_CQMEM_SIZE_LOW);
    DmemSize = ControlMemory->Read32(ALMAIF_INFO_DMEM_SIZE_LOW);

    ImemStart = ControlMemory->Read64(ALMAIF_INFO_IMEM_START_LOW);
    CQStart = ControlMemory->Read64(ALMAIF_INFO_CQMEM_START_LOW);
    DmemStart = ControlMemory->Read64(ALMAIF_INFO_DMEM_START_LOW);

    if (RelativeAddressing) {
      POCL_MSG_PRINT_ALMAIF("Almaif: Enabled relative addressing\n");
      CQStart += ControlMemory->PhysAddress();
      ImemStart += ControlMemory->PhysAddress();
      DmemStart += ControlMemory->PhysAddress();
    }

  } else {
    POCL_ABORT_UNIMPLEMENTED("Unsupported AlmaIF version\n");
  }
  POCL_MSG_PRINT_ALMAIF("CQStart=%p ImemStart=%p DmemStart=%p\n",
                        (void *)CQStart, (void *)ImemStart, (void *)DmemStart);
  POCL_MSG_PRINT_ALMAIF("CQSize=%u ImemSize=%u DmemSize=%u\n", CQSize, ImemSize,
                        DmemSize);
  POCL_MSG_PRINT_ALMAIF("ControlMemory->PhysAddress=%zu\n",
                        ControlMemory->PhysAddress());
  AllocRegions = (memory_region_t *)calloc(1, sizeof(memory_region_t));
  pocl_init_mem_region(AllocRegions,
                       DmemStart + ALMAIF_DEFAULT_CONSTANT_MEM_SIZE,
                       DmemSize - ALMAIF_DEFAULT_CONSTANT_MEM_SIZE);
  POCL_MSG_PRINT_ALMAIF(
      "Reserved %d bytes at the start of global memory for constant data\n",
      ALMAIF_DEFAULT_CONSTANT_MEM_SIZE);
}

void AlmaIFDevice::loadProgramToDevice(almaif_kernel_data_s *KernelData,
                                       cl_kernel Kernel,
                                       _cl_command_node *Command) {
  assert(KernelData);

  if (KernelData->imem_img_size == 0) {
    char img_file[POCL_MAX_PATHNAME_LENGTH];
    char cachedir[POCL_MAX_PATHNAME_LENGTH];
    // first try specialized
    pocl_cache_kernel_cachedir_path(img_file, Kernel->program,
                                    Command->program_device_i, Kernel,
                                    "/parallel.img", Command, 1);
    if (pocl_exists(img_file)) {
      pocl_cache_kernel_cachedir_path(cachedir, Kernel->program,
                                      Command->program_device_i, Kernel, "",
                                      Command, 1);
      prereadImages(cachedir, KernelData);
    } else {
      // if it doesn't exist, try specialized with local sizes 0-0-0
      // should pick either 0-0-0 or 0-0-0-goffs0
      _cl_command_node cmd_copy;
      memcpy(&cmd_copy, Command, sizeof(_cl_command_node));
      cmd_copy.command.run.pc.local_size[0] = 0;
      cmd_copy.command.run.pc.local_size[1] = 0;
      cmd_copy.command.run.pc.local_size[2] = 0;

      pocl_cache_kernel_cachedir_path(img_file, Kernel->program,
                                      Command->program_device_i, Kernel,
                                      "/parallel.img", &cmd_copy, 1);
      if (pocl_exists(img_file)) {
        pocl_cache_kernel_cachedir_path(cachedir, Kernel->program,
                                        Command->program_device_i, Kernel, "",
                                        &cmd_copy, 1);
      } else {
        pocl_cache_kernel_cachedir_path(cachedir, Kernel->program,
                                        Command->program_device_i, Kernel, "",
                                        &cmd_copy, 0);
      }
      POCL_MSG_PRINT_ALMAIF("Specialized kernel not found, using %s\n",
                           cachedir);
      prereadImages(cachedir, KernelData);
    }
  }

  assert(KernelData->imem_img_size > 0);

  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_RESET_CMD);

  InstructionMemory->CopyToMMAP(InstructionMemory->PhysAddress(),
                                KernelData->imem_img,
                                KernelData->imem_img_size);
  POCL_MSG_PRINT_ALMAIF("IMEM image written: %zu / %zu B\n",
                        InstructionMemory->PhysAddress(),
                        KernelData->imem_img_size);

  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_CONTINUE_CMD);
  HwClockStart = pocl_gettimemono_ns();
}

void AlmaIFDevice::prereadImages(const std::string &KernelCacheDir,
                                 almaif_kernel_data_s *KernelData) {
  POCL_MSG_PRINT_ALMAIF("Reading image files\n");
  uint64_t temp = 0;
  size_t size = 0;
  char *content = NULL;

  std::string module_fn = KernelCacheDir + "/parallel.img";

  if (pocl_exists(module_fn.c_str())) {
    int res = pocl_read_file(module_fn.c_str(), &content, &temp);
    size = (size_t)temp;
    assert(res == 0);
    assert(size > 0);
    assert(size < InstructionMemory->Size());
    KernelData->imem_img = content;
    KernelData->imem_img_size = size;
    content = NULL;
  } else
    POCL_ABORT("ALMAIF: %s for this kernel does not exist.\n",
               module_fn.c_str());

  module_fn = KernelCacheDir + "/kernel_address.txt";
  if (pocl_exists(module_fn.c_str())) {
    int res = pocl_read_file(module_fn.c_str(), &content, &temp);
    assert(res == 0);
    size = (size_t)temp;
    assert(size > 0);

    uint32_t kernel_address = 0;
    sscanf(content, "kernel address = %u", &kernel_address);
    assert(kernel_address != 0);
    KernelData->kernel_address = kernel_address;
    content = NULL;
  } else
    POCL_ABORT("ALMAIF: %s for this kernel does not exist.\n",
               module_fn.c_str());

  /*  snprintf(module_fn, POCL_MAX_PATHNAME_LENGTH, "%s/parallel_local.img",
             KernelCacheDir);
    if (pocl_exists(module_fn)) {
      int res = pocl_read_file(module_fn, &content, &temp);
      assert(res == 0);
      size = (size_t)temp;
      if (size == 0)
        POCL_MEM_FREE(content);
      KernelData->dmem_img = content;
      KernelData->dmem_img_size = size;

      uint32_t kernel_addr = 0;
      if (size) {
        void *p = content + size - 4;
        uint32_t *up = (uint32_t *)p;
        kernel_addr = *up;
     }
      POCL_MSG_PRINT_ALMAIF("Kernel address (%0x) found\n", kernel_addr);
      KernelData->kernel_address = kernel_addr;
      content = NULL;
    } else
      POCL_ABORT("ALMAIF: %s for this kernel does not exist.\n", module_fn);
  */
}

void AlmaIFDevice::printMemoryDump() {
  for (unsigned k = 0; k < InstructionMemory->Size(); k += 4) {
    uint32_t value = InstructionMemory->Read32(k);
    std::cerr << "IMEM at " << k << "=" << value << "\n";
  }
  for (unsigned k = 0; k < CQMemory->Size(); k += 4) {
    uint32_t value = CQMemory->Read32(k);
    std::cerr << "CQ at " << k << "=" << value << "\n";
  }

  for (unsigned k = 0; k < DataMemory->Size(); k += 4) {
    uint32_t value = DataMemory->Read32(k);
    std::cerr << "Data at " << k << "=" << value << "\n";
  }
  std::cerr << std::endl;
}

void AlmaIFDevice::writeDataToDevice(pocl_mem_identifier *DstMemId,
                                     const char *__restrict__ const Src,
                                     size_t Size, size_t Offset) {
  chunk_info_t *chunk = (chunk_info_t *)DstMemId->mem_ptr;
  size_t Dst = chunk->start_address + Offset;

  if (DataMemory->isInRange(Dst)) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes to 0x%zx\n", Size, Dst);
    DataMemory->CopyToMMAP(Dst, Src, Size);
  } else if (ExternalMemory && ExternalMemory->isInRange(Dst)) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes to external 0x%zx\n", Size,
                          Dst);
    ExternalMemory->CopyToMMAP(Dst, Src, Size);
  } else {
    POCL_ABORT(
        "Attempt to write data to outside the device memories. Address=%zu\n",
        Dst);
  }
}

void AlmaIFDevice::readDataFromDevice(char *__restrict__ const Dst,
                                      pocl_mem_identifier *SrcMemId,
                                      size_t Size, size_t Offset) {

  chunk_info_t *chunk = (chunk_info_t *)SrcMemId->mem_ptr;
  POCL_MSG_PRINT_ALMAIF("Reading data with chunk start %zu, and offset %zu\n",
                        chunk->start_address, Offset);
  size_t Src = chunk->start_address + Offset;
  if (DataMemory->isInRange(Src)) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes from 0x%zx\n", Size, Src);
    DataMemory->CopyFromMMAP(Dst, Src, Size);
  } else if (ExternalMemory && ExternalMemory->isInRange(Src)) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes from external 0x%zx\n",
                          Size, Src);
    ExternalMemory->CopyFromMMAP(Dst, Src, Size);
  } else {
    POCL_ABORT(
        "Attempt to read data from outside the device memories. Address=%zu\n",
        Src);
  }
}

size_t AlmaIFDevice::pointerDeviceOffset(pocl_mem_identifier *P) {
  assert(P->extra == 0);
  chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;
  assert(chunk != NULL);
  return chunk->start_address;
}

void AlmaIFDevice::freeBuffer(pocl_mem_identifier *P) {
  chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;

  POCL_MSG_PRINT_MEMORY("almaif: freed buffer from 0x%zx\n",
                        chunk->start_address);

  assert(chunk != NULL);
  pocl_free_chunk(chunk);
}

cl_int AlmaIFDevice::allocateBuffer(pocl_mem_identifier *P, size_t Size) {

  assert(P->mem_ptr == NULL);
  chunk_info_t *chunk = NULL;

  chunk = pocl_alloc_buffer(AllocRegions, Size);
  if (chunk == NULL)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  POCL_MSG_PRINT_MEMORY("almaif: allocated %zu bytes from 0x%zx\n", Size,
                        chunk->start_address);

  P->mem_ptr = chunk;
  P->version = 0;
  P->extra = 0;
  return CL_SUCCESS;
}
