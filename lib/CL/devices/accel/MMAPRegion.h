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

#ifndef MMAPREGION_H
#define MMAPREGION_H

#include <stdlib.h>

#include "pocl_types.h"

// MMAPRegion debug prints get quite spammy
// #define ACCEL_MMAP_DEBUG

class MMAPRegion
{
public:
  MMAPRegion ();
  MMAPRegion (size_t Address, size_t RegionSize, int mem_fd);
  virtual ~MMAPRegion ();

  virtual uint32_t Read32 (size_t offset);
  virtual void Write32 (size_t offset, uint32_t value);
  virtual void Write16 (size_t offset, uint16_t value);
  virtual uint64_t Read64 (size_t offset);

  virtual size_t VirtualToPhysical (void *ptr);

  virtual void CopyToMMAP (size_t destination, const void *source,
                           size_t bytes);
  virtual void CopyFromMMAP (void *destination, size_t source, size_t bytes);

  size_t PhysAddress;
  size_t Size;

protected:
  void *Data;
};

#endif
