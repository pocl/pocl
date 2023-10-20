/* AlmaIFDevice.hh - Interface class for accessing AlmaIF device.
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

#ifndef AlmaIFDevice_H
#define AlmaIFDevice_H

#include "AlmaIFRegion.hh"

#include "bufalloc.h"
#include "pocl_types.h"

#include <stdlib.h>
#include <string>

struct almaif_kernel_data_s;

class AlmaIFDevice {
public:
  AlmaIFDevice();
  virtual ~AlmaIFDevice();

  virtual void loadProgramToDevice(almaif_kernel_data_s *KernelData,
                                   cl_kernel Kernel, _cl_command_node *Command);

  AlmaIFRegion *ControlMemory;
  AlmaIFRegion *InstructionMemory;
  AlmaIFRegion *CQMemory;
  AlmaIFRegion *DataMemory;
  AlmaIFRegion *ExternalMemory = nullptr;
  memory_region_t *AllocRegions;

  bool RelativeAddressing;
  // if set to true, the device has its own time source in hardware,
  // and timestamps we get from device should be converted to host clock
  // for Emul it's false because we can get usable host timestamps in driver
  bool HasHardwareClock = false;
  // the *host* timestamp recorded when the time source on device is zero
  uint64_t HwClockStart = 0;
  // Hz, frequency of the device's clock source
  uint64_t HwClockFrequency = 0;
  int PointerSize = 0;

  void printMemoryDump();

  virtual void writeDataToDevice(pocl_mem_identifier *DstMemId,
                                 const char *__restrict__ const Src,
                                 size_t Size, size_t Offset);
  virtual void readDataFromDevice(char *__restrict__ const Dst,
                                  pocl_mem_identifier *SrcMemId, size_t Size,
                                  size_t Offset);

  virtual void discoverDeviceParameters();

  virtual bool isDBDevice() { return false; }

  // Allocate buffer from AlmaIFDevice's DataMemory or ExternalMemory
  virtual cl_int allocateBuffer(pocl_mem_identifier *P, size_t Size);
  // Free buffer from AlmaIFDevice's DataMemory or ExternalMemory
  virtual void freeBuffer(pocl_mem_identifier *P);
  // Retuns the offset of the allocated buffer, to be used as a kernel argument
  virtual size_t pointerDeviceOffset(pocl_mem_identifier *P);

protected:
  uintptr_t ImemStart;
  uint32_t ImemSize;
  uintptr_t CQStart;
  uint32_t CQSize;
  uintptr_t DmemStart;
  uint32_t DmemSize;

private:
  void prereadImages(const std::string &KernelCacheDir,
                     almaif_kernel_data_s *KernelData);
};

#endif
