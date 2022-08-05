/* XrtRegion.cc - accessing accelerator memory as memory mapped region.

   Copyright (c) 2019-2021 Pekka Jääskeläinen / Tampere University

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
//#include <stdio.h>
#include <fstream>

#include "experimental/xrt_kernel.h"

#include "XrtRegion.h"
#include "pocl_util.h"


XrtRegion::XrtRegion(size_t Address, size_t RegionSize, void *kernel) {

  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Initializing XrtRegion with Address %zu "
                      "and Size %zu and kernel %p\n",
                      Address, RegionSize, kernel);
  PhysAddress = Address;
  Size = RegionSize;
  Kernel = kernel;
  assert(Kernel != XRT_NULL_HANDLE &&
         "xrtKernelHandle NULL, is the kernel opened properly?");
}

XrtRegion::XrtRegion(size_t Address, size_t RegionSize, void *kernel,
                             char *init_file)
    : XrtRegion(Address, RegionSize, kernel) {

  if (RegionSize == 0) {
    return; // don't try to write to empty region
  }
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Initializing XrtRegion with file %s\n",
                      init_file);
  std::ifstream inFile;
  inFile.open(init_file, std::ios::binary);
  unsigned int current;
  int i = 0;
  while (inFile.good()) {
    inFile.read(reinterpret_cast<char *>(&current), sizeof(current));

    ((xrt::kernel *)Kernel)->write_register(Address + i, current);
    i += 4;
  }

  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Initialized region with %i bytes \n", i - 4);
}


uint32_t XrtRegion::Read32(size_t offset) {
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Reading from physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
  assert(Kernel != XRT_NULL_HANDLE && "No kernel handle; read before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  uint32_t value = ((xrt::kernel *)Kernel)->read_register(PhysAddress + offset);
  return value;
}

void XrtRegion::Write32(size_t offset, uint32_t value) {
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
  assert(Kernel != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  ((xrt::kernel *)Kernel)->write_register(PhysAddress + offset, value);
}

void XrtRegion::Write16(size_t offset, uint16_t value) {
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
  assert(Kernel != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");

  uint32_t old_value = ((xrt::kernel *)Kernel)
                           ->read_register(PhysAddress + (offset & 0xFFFFFFFC));

  uint32_t new_value = 0;
  if ((offset & 0b10) == 0) {
    new_value = (old_value & 0xFFFF0000) | (uint32_t)value;
  } else {
    new_value = ((uint32_t)value << 16) | (old_value & 0xFFFF);
  }
  ((xrt::kernel *)Kernel)
      ->write_register(PhysAddress + (offset & 0xFFFFFFFC), new_value);
}

uint64_t XrtRegion::Read64(size_t offset) {
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Reading from physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
  assert(Kernel != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  uint32_t value_low =
      ((xrt::kernel *)Kernel)->read_register(PhysAddress + offset);
  uint32_t value_high =
      ((xrt::kernel *)Kernel)->read_register(PhysAddress + offset + 4);
  uint64_t value = ((uint64_t)value_high << 32) | value_low;
  return value;
}


void XrtRegion::CopyToMMAP(size_t destination, const void *source,
                               size_t bytes) {
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Writing 0x%zx bytes to buffer at 0x%zx with "
                      "address 0x%zx\n",
                      bytes, PhysAddress, destination);
  auto src = (uint32_t *)source;
  size_t offset = destination - PhysAddress;
  assert(offset < Size && "Attempt to access data outside XRT memory");

  assert((offset & 0b11) == 0 &&
         "Xrt copytommap destination must be 4 byte aligned");
  assert(((size_t)src & 0b11) == 0 &&
         "Xrt copytommap source must be 4 byte aligned");
  assert((bytes % 4) == 0 && "Xrt copytommap size must be 4 byte multiple");

  for (size_t i = 0; i < bytes / 4; ++i) {
    ((xrt::kernel *)Kernel)->write_register(destination + 4 * i, src[i]);
  }
}

void XrtRegion::CopyFromMMAP(void *destination, size_t source,
                                 size_t bytes) {
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Reading 0x%zx bytes from buffer at 0x%zx "
                      "with address 0x%zx\n",
                      bytes, PhysAddress, source);
  auto dst = (uint32_t *)destination;
  size_t offset = source - PhysAddress;
  assert(offset < Size && "Attempt to access data outside XRT memory");

  assert((offset & 0b11) == 0 &&
         "Xrt copyfrommmap source must be 4 byte aligned");
  assert(((size_t)dst & 0b11) == 0 &&
         "Xrt copyfrommmap destination must be 4 byte aligned");
  assert((bytes % 4) == 0 && "Xrt copyfrommmap size must be 4 byte multiple");

  for (size_t i = 0; i < bytes / 4; ++i) {
    dst[i] = ((xrt::kernel *)Kernel)->read_register(source + 4 * i);
  }
}

void XrtRegion::CopyInMem (size_t source, size_t destination, size_t bytes) {
  POCL_MSG_PRINT_ACCEL_MMAP("XRTMMAP: Copying 0x%zx bytes from 0x%zx "
                      "to 0x%zx\n",
                      bytes, source, destination);
  size_t src_offset = source - PhysAddress;
  size_t dst_offset = destination - PhysAddress;
  assert(src_offset < Size && (src_offset+bytes) <= Size && "Attempt to access data outside XRT memory");
  assert(dst_offset < Size && (dst_offset+bytes) <= Size && "Attempt to access data outside XRT memory");
  assert((bytes % 4) == 0 && "Xrt copyinmem size must be 4 byte multiple");
  xrt::kernel *k = (xrt::kernel *)Kernel;

  for (size_t i = 0; i < bytes / 4; ++i) {
    uint32_t m = k->read_register(source + 4 * i);
    k->write_register(destination + 4 * i, m);
  }
}
