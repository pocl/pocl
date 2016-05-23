/* OpenCL built-in library: prefetch()

   Copyright (c) 2016 James Price / University of Bristol

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

/*
  The default implementation for prefetch is a no-op.

  Device specific backends should override this with something else if
  their hardware supports software prefetching.
*/

#define IMPLEMENT_PREFETCH_FUNCS_SINGLE(GENTYPE)                        \
  __attribute__((overloadable))                                         \
  void prefetch(const __global GENTYPE *p, size_t num_gentypes)         \
  {                                                                     \
  }

#define IMPLEMENT_PREFETCH_FUNCS(GENTYPE)             \
  IMPLEMENT_PREFETCH_FUNCS_SINGLE(GENTYPE)            \
  IMPLEMENT_PREFETCH_FUNCS_SINGLE(GENTYPE##2)         \
  IMPLEMENT_PREFETCH_FUNCS_SINGLE(GENTYPE##3)         \
  IMPLEMENT_PREFETCH_FUNCS_SINGLE(GENTYPE##4)         \
  IMPLEMENT_PREFETCH_FUNCS_SINGLE(GENTYPE##8)         \
  IMPLEMENT_PREFETCH_FUNCS_SINGLE(GENTYPE##16)

IMPLEMENT_PREFETCH_FUNCS(char);
IMPLEMENT_PREFETCH_FUNCS(uchar);
IMPLEMENT_PREFETCH_FUNCS(short);
IMPLEMENT_PREFETCH_FUNCS(ushort);
IMPLEMENT_PREFETCH_FUNCS(int);
IMPLEMENT_PREFETCH_FUNCS(uint);
__IF_INT64(IMPLEMENT_PREFETCH_FUNCS(long));
__IF_INT64(IMPLEMENT_PREFETCH_FUNCS(ulong));

__IF_FP16(IMPLEMENT_PREFETCH_FUNCS(half));
IMPLEMENT_PREFETCH_FUNCS(float);
__IF_FP64(IMPLEMENT_PREFETCH_FUNCS(double));
