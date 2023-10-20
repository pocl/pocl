/* XilinxXrtDevice.hh - Access AlmaIF device in Xilinx PCIe FPGA.

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

#ifndef XILINXXRTDEVICE_H
#define XILINXXRTDEVICE_H

#include "AlmaIFDevice.hh"

class XilinxXrtExternalRegion;

// This class abstracts the Almaif device instantiated on a Xilinx (PCIe) FPGA.
// The FPGA is reconfigured and Almaif device's memory map accessed with
// the Xilinx Runtime (XRT) API.
class XilinxXrtDevice : public AlmaIFDevice {
public:
  XilinxXrtDevice(const std::string &XrtKernelNamePrefix, unsigned j);
  XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                  const std::string &XclbinFile, unsigned j);
  XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                  const std::string &XclbinFile,
                  const std::string &ExternalMemoryParameters, unsigned j);
  void init_xrtdevice(const std::string &XrtKernelNamePrefix,
                      const std::string &XclbinFile,
                      const std::string &ExternalMemoryParameters, unsigned j);
  ~XilinxXrtDevice() override;
  // Reconfigures the FPGA
  void programBitstream(const std::string &XrtKernelNamePrefix,
                        const std::string &XclbinFile, unsigned j);

  // Allocate buffers from either on-chip or external memory regions
  // (Directs to either XilinxXrtRegion or XilinxXrtExternalRegion)
  cl_int allocateBuffer(pocl_mem_identifier *P, size_t Size) override;
  void freeBuffer(pocl_mem_identifier *P) override;
  // Retuns the offset of the allocated buffer, in order to be passed
  // as a kernel argument. This is relevant for XilinxXrtDevice specifically,
  // since the allocations in XilinxXrtExternalRegion are managed by XRT API.
  size_t pointerDeviceOffset(pocl_mem_identifier *P) override;
  void writeDataToDevice(pocl_mem_identifier *DstMemId,
                         const char *__restrict__ const Src, size_t Size,
                         size_t Offset) override;
  void readDataFromDevice(char *__restrict__ const Dst,
                          pocl_mem_identifier *SrcMemId, size_t Size,
                          size_t Offset) override;

private:
  XilinxXrtExternalRegion *ExternalXRTMemory;
  void *Kernel;
  int XilinxXrtDeviceInitDone_ = 0;
};

#endif
