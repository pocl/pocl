/* TTASimRegion.cc - TTASim device pretending to be mmapped device

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


#include "TTASimRegion.hh"

#include <Memory.hh>

#include "pocl_util.h"

#include <assert.h>

TTASimRegion::TTASimRegion(size_t Address, size_t RegionSize,
                           MemorySystem::MemoryPtr mem) {

  POCL_MSG_PRINT_ALMAIF_MMAP(
      "TTASim: Initializing TTASimRegion with Address %zu "
      "and Size %zu and memptr %p\n",
      Address, RegionSize, (void*)mem.get());
  PhysAddress_ = Address;
  Size_ = RegionSize;
  mem_ = mem;
  assert(mem != nullptr && "memory handle NULL, is the sim opened properly?");
}

uint32_t TTASimRegion::Read32(size_t offset) {

  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Reading from physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(mem_ != nullptr && "No memory handle; read before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");

  uint64_t result = 0;
  mem_->read(PhysAddress_ + offset, 4, result);
  return result;
}

void TTASimRegion::Write32(size_t offset, uint32_t value) {

  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Writing to physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(mem_ != nullptr && "No memory handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  mem_->writeDirectlyLE(PhysAddress_ + offset, 4, value);
}

void TTASimRegion::Write16(size_t offset, uint16_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Writing to physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(mem_ != nullptr && "No memory handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");

  mem_->writeDirectlyLE(PhysAddress_ + offset, 2, value);
}

uint64_t TTASimRegion::Read64(size_t offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Reading from physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);

  assert(mem_ != nullptr && "No memory handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");

  uint64_t result = 0;
  mem_->read(PhysAddress_ + offset, 8, result);
  return result;
}

void TTASimRegion::Write64(size_t offset, uint64_t value) {

  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Writing to physical address 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_, offset);
  assert(mem_ != nullptr && "No memory handle; write before mapping?");
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  mem_->writeDirectlyLE(PhysAddress_ + offset, 8, value);
}

void TTASimRegion::CopyToMMAP(size_t destination, const void *source,
                              size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "TTASim: Writing 0x%zx bytes to buffer at 0x%zx with "
      "address 0x%zx\n",
      bytes, PhysAddress_, destination);
  auto src = (uint8_t *)source;
  size_t offset = destination - PhysAddress_;
  assert(offset < Size_ && "Attempt to access data outside TTASim Region");

  for (size_t i = 0; i < bytes; ++i) {
    mem_->writeDirectlyLE(destination + i, 1, (Memory::MAU)src[i]);
  }
}

void TTASimRegion::CopyFromMMAP(void *destination, size_t source,
                                size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Reading 0x%zx bytes from buffer at 0x%zx "
                             "with address 0x%zx\n",
                             bytes, PhysAddress_, source);
  auto dst = (uint8_t *)destination;
  size_t offset = source - PhysAddress_;
  assert(offset < Size_ && "Attempt to access data outside TTASim Region");

  for (size_t i = 0; i < bytes; ++i) {
    dst[i] = mem_->read(source + i);
  }
}

void TTASimRegion::CopyInMem(size_t source, size_t destination, size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("TTASim: Copying 0x%zx bytes from 0x%zx "
                            "to 0x%zx\n",
                            bytes, source, destination);
  size_t src_offset = source - PhysAddress_;
  size_t dst_offset = destination - PhysAddress_;
  assert(src_offset < Size_ && (src_offset + bytes) <= Size_ &&
         "Attempt to access data outside TTASim Region");
  assert(dst_offset < Size_ && (dst_offset + bytes) <= Size_ &&
         "Attempt to access data outside TTASim Region");
  for (size_t i = 0; i < bytes; ++i) {
    Memory::MAU m = mem_->read(source + i);
    mem_->writeDirectlyLE(destination + i, 1, m);
  }
}
