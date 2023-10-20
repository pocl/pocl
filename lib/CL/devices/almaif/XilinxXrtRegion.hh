/* XilinxXrtRegion.hh - Access on-chip memory of an XRT device as AlmaIFRegion

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

#ifndef XILINXXRTREGION_H
#define XILINXXRTREGION_H

#include <stdlib.h>

#include "pocl_types.h"

#include "AlmaIFRegion.hh"

// Uses XRT's kernel IP API to abstract the on-chip memory
// of an Almaif device
class XilinxXrtRegion : public AlmaIFRegion {
public:
  XilinxXrtRegion(size_t Address, size_t RegionSize, void *kernel,
                  size_t DeviceOffset);
  XilinxXrtRegion(size_t Address, size_t RegionSize, void *kernel,
                  const std::string &init_file, size_t DeviceOffset);

  uint32_t Read32(size_t offset) override;
  void Write32(size_t offset, uint32_t value) override;
  void Write16(size_t offset, uint16_t value) override;
  uint64_t Read64(size_t offset) override;
  void Write64(size_t offset, uint64_t value) override;

  void CopyToMMAP(size_t destination, const void *source,
                  size_t bytes) override;
  void CopyFromMMAP(void *destination, size_t source, size_t bytes) override;
  void CopyInMem(size_t source, size_t destination, size_t bytes) override;

  void initRegion(const std::string &init_file);
  void setKernelPtr(void *ptr);

private:
  void *Kernel_;
  size_t DeviceOffset_;
};

#endif
