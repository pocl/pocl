/* XilinxXrtRegion.cc - Access on-chip memory of an XRT device as AlmaIFRegion

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

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
// #include <stdio.h>
#include <fstream>

#include "experimental/xrt_ip.h"

#include "XilinxXrtRegion.hh"
#include "pocl_util.h"

XilinxXrtRegion::XilinxXrtRegion(size_t Address, size_t RegionSize,
                                 void *kernel, size_t DeviceOffset) {

  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Initializing XilinxXrtRegion with Address %zu "
      "and Size %zu and kernel %p and DeviceOffset 0x%zx\n",
      Address, RegionSize, kernel, DeviceOffset);
  PhysAddress_ = Address;
  Size_ = RegionSize;
  Kernel_ = kernel;
  DeviceOffset_ = DeviceOffset;
  assert(Kernel_ != XRT_NULL_HANDLE &&
         "xrtKernelHandle NULL, is the kernel opened properly?");
}

XilinxXrtRegion::XilinxXrtRegion(size_t Address, size_t RegionSize,
                                 void *kernel, const std::string &init_file,
                                 size_t DeviceOffset)
    : XilinxXrtRegion(Address, RegionSize, kernel, DeviceOffset) {

  if (RegionSize == 0) {
    return; // don't try to write to empty region
  }
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Initializing XilinxXrtRegion with file %s\n",
      init_file.c_str());
  std::ifstream inFile;
  inFile.open(init_file, std::ios::binary);
  unsigned int current;
  int i = 0;
  while (inFile.good()) {
    inFile.read(reinterpret_cast<char *>(&current), sizeof(current));

    ((xrt::ip *)Kernel_)->write_register(Address + i - DeviceOffset_, current);
    i += 4;
  }

  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Initialized region with %i bytes \n",
                             i - 4);
}

void XilinxXrtRegion::initRegion(const std::string &init_file) {
  std::ifstream inFile;
  inFile.open(init_file, std::ios::binary);
  unsigned int current;
  int i = 0;
  while (inFile.good()) {
    inFile.read(reinterpret_cast<char *>(&current), sizeof(current));
    Write32(i, current);
    i += 4;
  }

  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Initialized region with %i bytes \n",
                             i - 4);
}

uint32_t XilinxXrtRegion::Read32(size_t offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Reading from region at 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_,
                             PhysAddress_ + offset - DeviceOffset_);
  assert(Kernel_ != XRT_NULL_HANDLE &&
         "No kernel handle; read before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  uint32_t value = ((xrt::ip *)Kernel_)
                       ->read_register(PhysAddress_ + offset - DeviceOffset_);
  return value;
}

void XilinxXrtRegion::Write32(size_t offset, uint32_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Writing to region at 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_,
                             PhysAddress_ + offset - DeviceOffset_);
  assert(Kernel_ != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  ((xrt::ip *)Kernel_)
      ->write_register(PhysAddress_ + offset - DeviceOffset_, value);
}

void XilinxXrtRegion::Write64(size_t offset, uint64_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Writing 64b to region at 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_,
                             PhysAddress_ + offset - DeviceOffset_);
  assert(Kernel_ != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  ((xrt::ip *)Kernel_)
      ->write_register(PhysAddress_ + offset - DeviceOffset_, value);
  ((xrt::ip *)Kernel_)
      ->write_register(PhysAddress_ + offset - DeviceOffset_ + 4, value >> 32);
}

void XilinxXrtRegion::Write16(size_t offset, uint16_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Writing 16b to region at 0x%zx with "
      "offset 0x%zx, DeviceOffset 0x%zx and total offset 0x%zx\n",
      PhysAddress_, offset, DeviceOffset_,
      PhysAddress_ + offset - DeviceOffset_);
  assert(Kernel_ != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");

  uint32_t old_value =
      ((xrt::ip *)Kernel_)
          ->read_register(PhysAddress_ + (offset & 0xFFFFFFFC) - DeviceOffset_);

  uint32_t new_value = 0;
  if ((offset & 0b10) == 0) {
    new_value = (old_value & 0xFFFF0000) | (uint32_t)value;
  } else {
    new_value = ((uint32_t)value << 16) | (old_value & 0xFFFF);
  }
  ((xrt::ip *)Kernel_)
      ->write_register(PhysAddress_ + (offset & 0xFFFFFFFC) - DeviceOffset_,
                       new_value);
}

uint64_t XilinxXrtRegion::Read64(size_t offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Reading 64b from region at 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_,
                             PhysAddress_ + offset - DeviceOffset_);
  assert(Kernel_ != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  uint32_t value_low =
      ((xrt::ip *)Kernel_)
          ->read_register(PhysAddress_ + offset - DeviceOffset_);
  uint32_t value_high =
      ((xrt::ip *)Kernel_)
          ->read_register(PhysAddress_ + offset - DeviceOffset_ + 4);
  uint64_t value = ((uint64_t)value_high << 32) | value_low;
  return value;
}

void XilinxXrtRegion::CopyToMMAP(size_t destination, const void *source,
                                 size_t bytes) {
  auto src = (uint32_t *)source;
  size_t offset = destination - PhysAddress_;
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Writing 0x%zx bytes to buffer at region 0x%zx with "
      "address 0x%zx and offset %zx\n",
      bytes, PhysAddress_, destination, offset);
  assert(offset < Size_ && "Attempt to access data outside XRT memory");

  assert((offset & 0b11) == 0 &&
         "Xrt copytommap destination must be 4 byte aligned");
  assert(((size_t)src & 0b11) == 0 &&
         "Xrt copytommap source must be 4 byte aligned");
  assert((bytes % 4) == 0 && "Xrt copytommap size must be 4 byte multiple");

  for (size_t i = 0; i < bytes / 4; ++i) {
    ((xrt::ip *)Kernel_)
        ->write_register(destination + 4 * i - DeviceOffset_, src[i]);
  }
}

void XilinxXrtRegion::CopyFromMMAP(void *destination, size_t source,
                                   size_t bytes) {
  auto dst = (uint32_t *)destination;
  size_t offset = source - PhysAddress_;
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Reading 0x%zx bytes from region at 0x%zx "
      "with address 0x%zx and offset\n",
      bytes, PhysAddress_, source, offset);
  assert(offset < Size_ && "Attempt to access data outside XRT memory");
  assert((offset & 0b11) == 0 &&
         "Xrt copyfrommmap source must be 4 byte aligned");

  switch (bytes) {
  case 1: {
    uint32_t value =
        ((xrt::ip *)Kernel_)->read_register(source - DeviceOffset_);
    *((uint8_t *)destination) = value;
    break;
  }
  case 2: {
    uint32_t value =
        ((xrt::ip *)Kernel_)->read_register(source - DeviceOffset_);
    *((uint16_t *)destination) = value;
    break;
  }
  default: {
    assert(((size_t)dst & 0b11) == 0 &&
           "Xrt copyfrommmap destination must be 4 byte aligned");
    size_t i;
    for (i = 0; i < bytes / 4; ++i) {
      dst[i] =
          ((xrt::ip *)Kernel_)->read_register(source - DeviceOffset_ + 4 * i);
    }
    if ((bytes % 4) != 0) {
      union value {
        char bytes[4];
        uint32_t full;
      } value1;
      value1.full =
          ((xrt::ip *)Kernel_)->read_register(source - DeviceOffset_ + 4 * i);
      for (int k = 0; k < (bytes % 4); k++) {
        dst[i] = value1.bytes[k];
      }
    }
  }
  }
}

void XilinxXrtRegion::CopyInMem(size_t source, size_t destination,
                                size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Copying 0x%zx bytes from 0x%zx "
                             "to 0x%zx\n",
                             bytes, source, destination);
  size_t src_offset = source - PhysAddress_;
  size_t dst_offset = destination - PhysAddress_;
  assert(src_offset < Size_ && (src_offset + bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
  assert(dst_offset < Size_ && (dst_offset + bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
  assert((bytes % 4) == 0 && "Xrt copyinmem size must be 4 byte multiple");
  xrt::ip *k = (xrt::ip *)Kernel_;

  for (size_t i = 0; i < bytes / 4; ++i) {
    uint32_t m = k->read_register(source - DeviceOffset_ + 4 * i);
    k->write_register(destination - DeviceOffset_ + 4 * i, m);
  }
}

void XilinxXrtRegion::setKernelPtr(void *ptr) { Kernel_ = ptr; }
