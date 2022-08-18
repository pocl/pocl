/* Device.hh - basic way of accessing accelerator memory.
 *                 as a memory mapped region

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

#ifndef Device_H
#define Device_H

#include "Region.hh"
#include "almaif-compile.hh"

#include "bufalloc.h"
#include "pocl_types.h"

#include <stdlib.h>

class Device {
public:
  Device();
  virtual ~Device();

  virtual void loadProgramToDevice(almaif_kernel_data_t *kd, cl_kernel kernel,
                                   _cl_command_node *cmd);

  Region *ControlMemory;
  Region *InstructionMemory;
  Region *CQMemory;
  Region *DataMemory;
  Region *ExternalMemory = nullptr;
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

protected:
  virtual void discoverDeviceParameters();
  uintptr_t imem_start;
  uint32_t imem_size;
  uintptr_t cq_start;
  uint32_t cq_size;
  uintptr_t dmem_start;
  uint32_t dmem_size;

private:
  void preread_images(const char *kernel_cachedir, almaif_kernel_data_t *kd);
};

#endif
