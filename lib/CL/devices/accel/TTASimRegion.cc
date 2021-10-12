


#include "TTASimRegion.h"


#include <Memory.hh>

#include "pocl_util.h"

#include <assert.h>

//#define ACCEL_TTASIM_DEBUG

TTASimRegion::TTASimRegion(size_t Address, size_t RegionSize, MemorySystem::MemoryPtr mem) {

#ifdef ACCEL_TTASIM_DEBUG
  POCL_MSG_PRINT_INFO("TTASim: Initializing TTASimRegion with Address %zu "
                      "and Size %zu and memptr %p\n",
                      Address, RegionSize, mem);
#endif
  PhysAddress = Address;
  Size = RegionSize;
  mem_ = mem;
  assert(mem != nullptr &&
         "memory handle NULL, is the sim opened properly?");
}
uint32_t TTASimRegion::Read32(size_t offset) {

#ifdef ACCEL_TTASIM_DEBUG
  POCL_MSG_PRINT_INFO("TTASim: Reading from physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
  assert(mem_ != nullptr && "No memory handle; read before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  
  uint64_t result = 0;
  mem_->read(PhysAddress + offset, 4, result);
  return result;
}

void TTASimRegion::Write32(size_t offset, uint32_t value) {

#ifdef ACCEL_TTASIM_DEBUG
  POCL_MSG_PRINT_INFO("TTASim: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
  assert(mem_ != nullptr &&
         "No memory handle; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");
  mem_->writeDirectlyLE((uint32_t)(PhysAddress + offset), 4, value);

}

void TTASimRegion::Write16(size_t offset, uint16_t value) {
#ifdef ACCEL_TTASIM_DEBUG
  POCL_MSG_PRINT_INFO("TTASim: Writing to physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif
  assert(mem_ != nullptr &&
         "No memory handle; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");

  mem_->writeDirectlyLE(PhysAddress + offset, 2, value);
}

uint64_t TTASimRegion::Read64(size_t offset) {
#ifdef ACCEL_TTASIM_DEBUG
  POCL_MSG_PRINT_INFO("TTASim: Reading from physical address 0x%zx with "
                      "offset 0x%zx\n",
                      PhysAddress, offset);
#endif

  assert(mem_ != nullptr &&
         "No memory handle; write before mapping?");
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");

  uint64_t result = 0;
  mem_->read(PhysAddress + offset, 8, result);
  return result;
}


void TTASimRegion::CopyToMMAP(size_t destination, const void *source,
                               size_t bytes) {
#ifdef ACCEL_TTASIM_DEBUG
  POCL_MSG_PRINT_INFO("TTASim: Writing 0x%zx bytes to buffer at 0x%zx with "
                      "address 0x%zx\n",
                      bytes, PhysAddress, destination);
#endif
  auto src = (uint8_t *)source;
  size_t offset = destination - PhysAddress;
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");

  for (size_t i = 0; i < bytes; ++i) {
    mem_->writeDirectlyLE(destination + i, 1, (Memory::MAU)src[i]);
  }
}

void TTASimRegion::CopyFromMMAP(void *destination, size_t source,
                                 size_t bytes) {
#ifdef ACCEL_TTASIM_DEBUG
  POCL_MSG_PRINT_INFO("TTASim: Reading 0x%zx bytes from buffer at 0x%zx "
                      "with address 0x%zx\n",
                      bytes, PhysAddress, source);
#endif
  auto dst = (uint8_t *)destination;
  size_t offset = source - PhysAddress;
  assert(offset < Size && "Attempt to access data outside MMAP'd buffer");

  for (size_t i = 0; i < bytes; ++i) {
    dst[i] = mem_->read(source + i);
  }
}


