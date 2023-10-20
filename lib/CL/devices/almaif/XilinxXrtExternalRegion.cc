/* XilinxXrtExternalRegion.cc - Access external memory (DDR or HBM) of an XRT
 device
 *                        as AlmaIFRegion

   Copyright (c) 2023 Topi Leppänen / Tampere University

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
#include <unistd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"

#include "XilinxXrtExternalRegion.hh"
#include "pocl_util.h"

XilinxXrtExternalRegion::XilinxXrtExternalRegion(size_t Address,
                                                 size_t RegionSize,
                                                 void *Device) {

  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Initializing XilinxXrtExternalRegion with Address %zu "
      "and Size %zu and device %p\n",
      Address, RegionSize, Device);
  PhysAddress_ = Address;
  Size_ = RegionSize;

  XilinxXrtDeviceHandle_ = Device;
}

void XilinxXrtExternalRegion::freeBuffer(pocl_mem_identifier *P) {
  delete (xrt::bo *)(P->mem_ptr);
  P->mem_ptr = NULL;
}

uint64_t XilinxXrtExternalRegion::pointerDeviceOffset(pocl_mem_identifier *P) {
  assert(P->mem_ptr);
  return ((xrt::bo *)(P->mem_ptr))->address();
}

// Buffer allocation uses XRT buffer allocation API
cl_int XilinxXrtExternalRegion::allocateBuffer(pocl_mem_identifier *P,
                                               size_t Size) {
  xrt::bo *DeviceBuffer = new xrt::bo(*(xrt::device *)XilinxXrtDeviceHandle_,
                                      Size, (xrt::memory_group)0);

  assert(DeviceBuffer != XRT_NULL_HANDLE && "xrtBufferHandle NULL");
  P->mem_ptr = DeviceBuffer;
  uint64_t PhysAddress = pointerDeviceOffset(P);
  POCL_MSG_PRINT_ALMAIF(
      "XRTMMAP: Initialized XilinxXrtExternalRegion buffer with "
      "physical address %" PRIu64 "\n",
      PhysAddress);
  return CL_SUCCESS;
}

void XilinxXrtExternalRegion::CopyToMMAP(pocl_mem_identifier *DstMemId,
                                         const void *Source, size_t Bytes,
                                         size_t Offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Writing 0x%zx bytes to buffer at 0x%zx with "
      "address 0x%zx\n",
      Bytes, PhysAddress_, pointerDeviceOffset(DstMemId));
  auto src = (uint32_t *)Source;
  assert(Offset < Size_ && "Attempt to access data outside XRT memory");

  xrt::bo *b = (xrt::bo *)(DstMemId->mem_ptr);
  assert(b != XRT_NULL_HANDLE && "No buffer handle?");
  b->write(Source, Bytes, Offset);
  b->sync(XCL_BO_SYNC_BO_TO_DEVICE, Bytes, Offset);
}

void XilinxXrtExternalRegion::CopyFromMMAP(void *Destination,
                                           pocl_mem_identifier *SrcMemId,
                                           size_t Bytes, size_t Offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Reading 0x%zx bytes from buffer at 0x%zx "
      "with address 0x%zx\n",
      Bytes, PhysAddress_, pointerDeviceOffset(SrcMemId));
  assert(Offset < Size_ && "Attempt to access data outside XRT memory");

  xrt::bo *b = (xrt::bo *)(SrcMemId->mem_ptr);
  assert(b != XRT_NULL_HANDLE && "No kernel handle?");
  b->sync(XCL_BO_SYNC_BO_FROM_DEVICE, Bytes, Offset);
  b->read(Destination, Bytes, Offset);
}

void XilinxXrtExternalRegion::CopyInMem(size_t Source, size_t Destination,
                                        size_t Bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Copying 0x%zx bytes from 0x%zx "
                             "to 0x%zx\n",
                             Bytes, Source, Destination);
  size_t SrcOffset = Source - PhysAddress_;
  size_t DstOffset = Destination - PhysAddress_;
  assert(SrcOffset < Size_ && (SrcOffset + Bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
  assert(DstOffset < Size_ && (DstOffset + Bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
//  assert(DeviceBuffer != XRT_NULL_HANDLE &&
//         "No kernel handle; write before mapping?");
/*
  xrt::bo *b = (xrt::bo *)DeviceBuffer;
  auto b_mapped = b->map();

  b->sync(XCL_BO_SYNC_BO_FROM_DEVICE, Bytes, SrcOffset);
  memcpy((char *)b_mapped + DstOffset, (char *)b_mapped + SrcOffset, Bytes);
  b->sync(XCL_BO_SYNC_BO_TO_DEVICE, Bytes, DstOffset);
*/}
