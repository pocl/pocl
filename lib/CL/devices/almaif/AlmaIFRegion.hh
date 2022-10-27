/* AlmaIFRegion.hh - Interface class for raw memory operations
 * (read and write operations to backend-specific memory)

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

#ifndef AlmaIFRegion_H
#define AlmaIFRegion_H

#include "pocl_types.h"

#include <stdlib.h>

class AlmaIFRegion {
public:
  virtual ~AlmaIFRegion();
  virtual uint32_t Read32(size_t offset) = 0;
  virtual void Write32(size_t offset, uint32_t value) = 0;
  virtual void Write16(size_t offset, uint16_t value) = 0;
  virtual uint64_t Read64(size_t offset) = 0;
  virtual void Write64(size_t offset, uint64_t value) = 0;

  virtual void CopyToMMAP(size_t destination, const void *source,
                          size_t bytes) = 0;
  virtual void CopyFromMMAP(void *destination, size_t source, size_t bytes) = 0;
  virtual void CopyInMem(size_t source, size_t destination, size_t bytes) = 0;

  virtual bool isInRange(size_t dst);

  size_t PhysAddress;
  size_t Size;
};

#endif
