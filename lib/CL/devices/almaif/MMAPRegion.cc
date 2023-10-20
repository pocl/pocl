/* MMAPRegion.cc - accessing accelerator memory as memory mapped region.

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
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#include "pocl_util.h"

#include "MMAPRegion.hh"

MMAPRegion::MMAPRegion() {}

MMAPRegion::MMAPRegion(size_t Address, size_t RegionSize, int mem_fd) {
  PhysAddress_ = Address;
  Size_ = RegionSize;
  if (Size_ == 0) {
    return;
  }
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "almaif: mmap'ing from address 0x%zx with size %zu\n", Address,
      RegionSize);
  // In case of unaligned Address, align the mmap call
  long page_size = sysconf(_SC_PAGESIZE);
  size_t roundDownAddress = (Address / page_size) * page_size;
  size_t difference = Address - roundDownAddress;
  Data = mmap(0, Size_ + difference, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd,
              roundDownAddress);
  assert(Data != MAP_FAILED && "MMAPRegion mapping failed");
  // Increment back to the unaligned address user asked for
  Data = (void *)((char *)Data + difference);
  POCL_MSG_PRINT_ALMAIF_MMAP("almaif: got address %p\n", Data);
}

void MMAPRegion::initRegion(const std::string &init_file) {
  std::ifstream inFile;
  inFile.open(init_file.c_str(), std::ios::binary);
  unsigned int current;
  int i = 0;
  while (inFile.good()) {
    inFile.read(reinterpret_cast<char *>(&current), sizeof(current));
    Write32(i, current);
    i += 4;
  }

  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Initialized region with %i bytes \n", i - 4);
}

MMAPRegion::~MMAPRegion() {
  POCL_MSG_PRINT_ALMAIF_MMAP("almaif: munmap'ing from address 0x%zx\n",
                             PhysAddress_);
  if (Data) {
    // Align unmap to page_size
    long page_size = sysconf(_SC_PAGESIZE);
    size_t roundDownAddress = ((size_t)Data / page_size) * page_size;
    size_t difference = (size_t)Data - roundDownAddress;

    munmap((void *)roundDownAddress, Size_ + difference);
    Data = NULL;
  }
}

uint32_t MMAPRegion::Read32(size_t offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Reading from physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(Data && "No pointer to MMAP'd region; read before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  auto value =
      static_cast<volatile uint32_t *>(Data)[offset / sizeof(uint32_t)];
  return value;
}

void MMAPRegion::Write32(size_t offset, uint32_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Writing to physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(Data && "No pointer to MMAP'd region; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  static_cast<volatile uint32_t *>(Data)[offset / sizeof(uint32_t)] = value;
}

void MMAPRegion::Write16(size_t offset, uint16_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Writing to physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(Data && "No pointer to MMAP'd region; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  static_cast<volatile uint16_t *>(Data)[offset / sizeof(uint16_t)] = value;
}

uint64_t MMAPRegion::Read64(size_t offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Reading from physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(Data && "No pointer to MMAP'd region; read before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  auto value =
      static_cast<volatile uint64_t *>(Data)[offset / sizeof(uint64_t)];
  return value;
}

void MMAPRegion::Write64(size_t offset, uint64_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Writing to physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(Data && "No pointer to MMAP'd region; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  static_cast<volatile uint64_t *>(Data)[offset / sizeof(uint64_t)] = value;
}

void MMAPRegion::CopyToMMAP(size_t destination, const void *source,
                            size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "MMAP: Writing 0x%zx bytes to buffer at 0x%zx with "
      "address 0x%zx\n",
      bytes, PhysAddress_, destination);
  auto src = (char *)source;
  size_t offset = destination - PhysAddress_;
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  auto dst = offset + static_cast<volatile char *>(Data);
  memcpy((void *)dst, src, bytes);
}

void MMAPRegion::CopyFromMMAP(void *destination, size_t source, size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Reading 0x%zx bytes from buffer at 0x%zx "
                             "with address 0x%zx\n",
                             bytes, PhysAddress_, source);
  auto dst = (char *)destination;
  size_t offset = source - PhysAddress_;
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  auto src = offset + static_cast<volatile char *>(Data);
  memcpy(dst, (void *)src, bytes);
}

void MMAPRegion::CopyInMem(size_t source, size_t destination, size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Copying 0x%zx bytes from 0x%zx "
                            "to 0x%zx\n",
                            bytes, source, destination);
  size_t src_offset = source - PhysAddress_;
  size_t dst_offset = destination - PhysAddress_;
  assert(src_offset < Size_ && (src_offset + bytes) <= Size_ &&
         "Attempt to access data outside MMAP'd buffer");
  assert(dst_offset < Size_ && (dst_offset + bytes) <= Size_ &&
         "Attempt to access data outside MMAP'd buffer");
  volatile char *src = src_offset + static_cast<volatile char *>(Data);
  volatile char *dst = dst_offset + static_cast<volatile char *>(Data);
  memcpy((void *)dst, (void *)src, bytes);
}
