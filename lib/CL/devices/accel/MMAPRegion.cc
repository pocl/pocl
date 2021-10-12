/* MMAPRegion.cc - accessing accelerator memory as memory mapped region.

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
#include <sys/mman.h>
#include <unistd.h>

#include "MMAPRegion.h"

// MMAPRegion debug prints get quite spammy
// #define ACCEL_MMAP_DEBUG

MMAPRegion::MMAPRegion(){}

MMAPRegion::MMAPRegion(size_t Address, size_t RegionSize, int mem_fd) {
  PhysAddress = Address;
  Size = RegionSize;
  if (Size == 0) {
    return;
  }
#ifdef ACCEL_MMAP_DEBUG
  POCL_MSG_PRINT_INFO("accel: mmap'ing from address 0x%zx with size %zu\n",
                      Address, RegionSize);
#endif
  // In case of unaligned Address, align the mmap call
  long page_size = sysconf(_SC_PAGESIZE);
  size_t roundDownAddress = (Address / page_size) * page_size;
  size_t difference = Address - roundDownAddress;
  Data = mmap(0, Size + difference, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd,
              roundDownAddress);
  assert(Data != MAP_FAILED && "MMAPRegion mapping failed");
  // Increment back to the unaligned address user asked for
  Data = (void *)((char *)Data + difference);
#ifdef ACCEL_MMAP_DEBUG
  POCL_MSG_PRINT_INFO("accel: got address %p\n", Data);
#endif
}

MMAPRegion::~MMAPRegion() {
#ifdef ACCEL_MMAP_DEBUG
  POCL_MSG_PRINT_INFO("accel: munmap'ing from address 0x%zx\n", PhysAddress);
#endif
  if (Data) {
    // Align unmap to page_size
    long page_size = sysconf(_SC_PAGESIZE);
    size_t roundDownAddress = ((size_t)Data / page_size) * page_size;
    size_t difference = (size_t)Data - roundDownAddress;

    munmap((void *)roundDownAddress, Size + difference);
    Data = NULL;
  }
}

uint32_t MMAPRegion::Read32(size_t offset) {
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

void MMAPRegion::Write32(size_t offset, uint32_t value) {
#ifdef ACCEL_MMAP_DEBUG
  POCL_MSG_PRINT_INFO("MMAP: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
  assert(Data && "No pointer to MMAP'd region; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  static_cast<volatile uint32_t *>(Data)[offset / sizeof(uint32_t)] = value;
}

void MMAPRegion::Write16(size_t offset, uint16_t value) {
#ifdef ACCEL_MMAP_DEBUG
  POCL_MSG_PRINT_INFO("MMAP: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
  assert(Data && "No pointer to MMAP'd region; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  static_cast<volatile uint16_t *>(Data)[offset / sizeof(uint16_t)] = value;
}

uint64_t MMAPRegion::Read64(size_t offset) {
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


void MMAPRegion::CopyToMMAP(size_t destination, const void *source,
                            size_t bytes) {
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

void MMAPRegion::CopyFromMMAP(void *destination, size_t source, size_t bytes) {
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
