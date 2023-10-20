/* XilinxXrtExternalRegion.hh - Access external memory (DDR or HBM) of an XRT
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

#ifndef POCL_XILINXXRTEXTERNALREGION_H
#define POCL_XILINXXRTEXTERNALREGION_H

#include <stdlib.h>

#include "pocl_cl.h"

class XilinxXrtExternalRegion {
public:
  XilinxXrtExternalRegion(size_t Address, size_t RegionSize, void *Device);

  void CopyToMMAP(pocl_mem_identifier *DstMemId, const void *Source,
                  size_t Bytes, size_t Offset);
  void CopyFromMMAP(void *Destination, pocl_mem_identifier *SrcMemId,
                    size_t Bytes, size_t Offset);
  void CopyInMem(size_t Source, size_t Destination, size_t Bytes);

  // Returns the offset of the allocated pointer in the Xrt address space
  // used by the kernel
  uint64_t pointerDeviceOffset(pocl_mem_identifier *P);
  // Buffer allocation uses XRT buffer allocation API.
  // This is done in order to support multiple distinct external memory
  // types in Xilinx PCIe FPGAs (multiple HBM and DDR banks).
  // The alternative of using our bufalloc library to map the entire memory
  // banks as bufalloc-regions was found to have significant performance
  // issues when buffers were being read and written via the XRT API.
  // (Possibly the entire bufalloc-regions were being read/flushed when only
  // parts of it were read or written, or something to that effect.)
  cl_int allocateBuffer(pocl_mem_identifier *P, size_t Size);
  void freeBuffer(pocl_mem_identifier *P);

private:
  size_t Size_;
  size_t PhysAddress_;
  void *XilinxXrtDeviceHandle_;
};

#endif
