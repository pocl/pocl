/* MMAPRegion.hh - basic way of accessing accelerator memory.
 *                 as a memory mapped region

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

#ifndef XRTMMAPREGION_H
#define XRTMMAPREGION_H

#include <stdlib.h>

#include "pocl_types.h"

#include "MMAPRegion.h"

class XrtMMAPRegion : public MMAPRegion
{
public:
  XrtMMAPRegion ();
  XrtMMAPRegion (size_t Address, size_t RegionSize, char *xrt_kernel_name);
  XrtMMAPRegion (size_t Address, size_t RegionSize, void *kernel);
  XrtMMAPRegion (size_t Address, size_t RegionSize, void *kernel,
                 char *init_file);

  ~XrtMMAPRegion ();

  uint32_t Read32 (size_t offset);
  void Write32 (size_t offset, uint32_t value);
  void Write16 (size_t offset, uint16_t value);
  uint64_t Read64 (size_t offset);

  size_t VirtualToPhysical (void *ptr);

  void CopyToMMAP (size_t destination, const void *source, size_t bytes);
  void CopyFromMMAP (void *destination, size_t source, size_t bytes);

  void *GetKernelHandle ();

private:
  void *Kernel;
  void *DeviceHandle;
  bool OpenedDevice = false;
};

#endif
