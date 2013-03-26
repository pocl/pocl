/* OpenCL built-in library: popcount()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   
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

#include "../templates.h"

/* These explicit implementations are taken from
   <http://aggregate.org/MAGIC/>:
   
   @techreport{magicalgorithms,
   author={Henry Gordon Dietz},
   title={{The Aggregate Magic Algorithms}},
   institution={University of Kentucky},
   howpublished={Aggregate.Org online technical report},
   date={2013-03-25},
   URL={http://aggregate.org/MAGIC/}
   }
*/

#define __builtin_popcount0uhh(n)               \
  ({                                            \
    uchar __n=(n);                              \
    __n -= (__n >> 1) & 0x55U;                  \
    __n = ((__n >> 2) & 0x33U) + (__n & 0x33U); \
    __n = ((__n >> 4) + __n) & 0x0fU;           \
    __n;                                        \
  })

#define __builtin_popcount0uh(n)                        \
  ({                                                    \
    ushort __n=(n);                                     \
    __n -= (__n >> 1) & 0x5555U;                        \
    __n = ((__n >> 2) & 0x3333U) + (__n & 0x3333U);     \
    __n = ((__n >> 4) + __n) & 0x0f0fU;                 \
    __n += __n >> 8;                                    \
    __n & 0x001fU;                                      \
  })

#define __builtin_popcount0u(n)                                 \
  ({                                                            \
    uint __n=(n);                                               \
    __n -= (__n >> 1) & 0x55555555U;                            \
    __n = ((__n >> 2) & 0x33333333U) + (__n & 0x33333333U);     \
    __n = ((__n >> 4) + __n) & 0x0f0f0f0fU;                     \
    __n += __n >> 8;                                            \
    __n += __n >> 16;                                           \
    __n & 0x0000003fU;                                          \
  })

#define __builtin_popcount0ul(n)                                        \
  ({                                                                    \
    ulong __n=(n);                                                      \
    __n -= (__n >> 1) & 0x5555555555555555UL;                           \
    __n = ((__n >> 2) & 0x3333333333333333UL) + (__n & 0x3333333333333333UL); \
    __n = ((__n >> 4) + __n) & 0x0f0f0f0f0f0f0f0fUL;                    \
    __n += __n >> 8;                                                    \
    __n += __n >> 16;                                                   \
    __n += __n >> 32;                                                   \
    __n & 0x000000000000007fUL;                                         \
  })

#define __builtin_popcount0hh(n) __builtin_popcount0uhh(n)
#define __builtin_popcount0h(n)  __builtin_popcount0uh(n)
#define __builtin_popcount0(n)   __builtin_popcount0u(n)
#define __builtin_popcount0l(n)  __builtin_popcount0ul(n)

#define popcount0 popcount
DEFINE_BUILTIN_G_G(popcount0)
#undef popcount0
