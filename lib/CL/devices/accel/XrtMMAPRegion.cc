/* XrtMMAPRegion.cc - accessing accelerator memory as memory mapped region.

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

#include "XrtMMAPRegion.h"
#include "pocl_util.h"

// XrtMMAPRegion debug prints get quite spammy
//#define ACCEL_XRTMMAP_DEBUG

XrtMMAPRegion::XrtMMAPRegion(size_t Address, size_t RegionSize,
                             char *xrt_kernel_name) {

#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO(
      "XRTMMAP: Initializing XrtMMAPregion with Address %zu and Size %zu\n",
      Address, RegionSize);
#endif
  char xclbin_char[120];
  snprintf(xclbin_char, sizeof(xclbin_char), "%s.xclbin", xrt_kernel_name);

  char kernel_char[120];
  snprintf(kernel_char, sizeof(kernel_char), "%s:{%s_1}", xrt_kernel_name,
           xrt_kernel_name);

  auto devicehandle = new xrt::device(0);
  assert(devicehandle != NULL && "devicehandle null\n");

  auto uuid = devicehandle->load_xclbin(xclbin_char);
  auto kernel = new xrt::kernel(*devicehandle, uuid, kernel_char,
                                xrt::kernel::cu_access_mode::exclusive);
  Kernel = (void *)kernel;
  DeviceHandle = (void *)devicehandle;

  PhysAddress = Address;
  Size = RegionSize;
  assert(kernel != XRT_NULL_HANDLE &&
         "xrtKernelHandle NULL, is the kernel opened properly?");

  OpenedDevice = true;
}

XrtMMAPRegion::XrtMMAPRegion(size_t Address, size_t RegionSize, void *kernel) {

#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Initializing XrtMMAPregion with Address %zu "
                      "and Size %zu and kernel %p\n",
                      Address, RegionSize, kernel);
#endif
  PhysAddress = Address;
  Size = RegionSize;
  Kernel = kernel;
  assert(Kernel != XRT_NULL_HANDLE &&
         "xrtKernelHandle NULL, is the kernel opened properly?");
}

XrtMMAPRegion::XrtMMAPRegion(size_t Address, size_t RegionSize, void *kernel,
                             char *init_file)
    : XrtMMAPRegion(Address, RegionSize, kernel) {

  if (RegionSize == 0) {
    return; // don't try to write to empty region
  }
#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Initializing XrtMMAPregion with file %s\n",
                      init_file);
#endif
  std::ifstream inFile;
  inFile.open(init_file, std::ios::binary);
  unsigned int current;
  int i = 0;
  while (inFile.good()) {
    inFile.read(reinterpret_cast<char *>(&current), sizeof(current));

    ((xrt::kernel *)Kernel)->write_register(Address + i, current);
    i += 4;
  }

#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Initialized region with %i bytes \n", i - 4);
#endif
}

XrtMMAPRegion::~XrtMMAPRegion() {
  if (OpenedDevice) {
    delete ((xrt::kernel *)Kernel);
    delete ((xrt::device *)DeviceHandle);
  }
}

uint32_t XrtMMAPRegion::Read32(size_t offset) {
#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Reading from physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
  assert(Kernel != XRT_NULL_HANDLE && "No kernel handle; read before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  uint32_t value = ((xrt::kernel *)Kernel)->read_register(PhysAddress + offset);
  return value;
}

void *XrtMMAPRegion::GetKernelHandle() { return Kernel; }

void XrtMMAPRegion::Write32(size_t offset, uint32_t value) {
#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
  assert(Kernel != XRT_NULL_HANDLE &&
         "No kernel handle; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  ((xrt::kernel *)Kernel)->write_register(PhysAddress + offset, value);
}

void XrtMMAPRegion::Write16(size_t offset, uint16_t value) {
#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
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

uint64_t XrtMMAPRegion::Read64(size_t offset) {
#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Reading from physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
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

size_t XrtMMAPRegion::VirtualToPhysical(void *ptr) {
  size_t offset = ((size_t)ptr) - PhysAddress;
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  return offset + PhysAddress;
}

void XrtMMAPRegion::CopyToMMAP(size_t destination, const void *source,
                               size_t bytes) {
#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Writing 0x%zx bytes to buffer at 0x%zx with "
                      "address 0x%zx\n",
                      bytes, PhysAddress, destination);
#endif
  auto src = (uint32_t *)source;
  size_t offset = destination - PhysAddress;
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");

  assert((offset & 0b11) == 0 &&
         "Xrt copytommap destination must be 4 byte aligned");
  assert(((size_t)src & 0b11) == 0 &&
         "Xrt copytommap source must be 4 byte aligned");
  assert((bytes % 4) == 0 && "Xrt copytommap size must be 4 byte multiple");

  for (size_t i = 0; i < bytes / 4; ++i) {
    ((xrt::kernel *)Kernel)->write_register(destination + 4 * i, src[i]);
  }
}

void XrtMMAPRegion::CopyFromMMAP(void *destination, size_t source,
                                 size_t bytes) {
#ifdef ACCEL_XRTMMAP_DEBUG
  POCL_MSG_PRINT_INFO("XRTMMAP: Reading 0x%zx bytes from buffer at 0x%zx "
                      "with address 0x%zx\n",
                      bytes, PhysAddress, source);
#endif
  auto dst = (uint32_t *)destination;
  size_t offset = source - PhysAddress;
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");

  assert((offset & 0b11) == 0 &&
         "Xrt copyfrommmap source must be 4 byte aligned");
  assert(((size_t)dst & 0b11) == 0 &&
         "Xrt copyfrommmap destination must be 4 byte aligned");
  assert((bytes % 4) == 0 && "Xrt copyfrommmap size must be 4 byte multiple");

  for (size_t i = 0; i < bytes / 4; ++i) {
    dst[i] = ((xrt::kernel *)Kernel)->read_register(source + 4 * i);
  }
}
