/* OpenCL built-in library: clz()

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

/* These implementations return 8*sizeof(TYPE) when the input is 0 */

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

#define __builtin_clz0uhh(n)                    \
  ({                                            \
    uchar __n=(n);                              \
    __n |= __n >> 1;                            \
    __n |= __n >> 2;                            \
    __n |= __n >> 4;                            \
    8 - popcount(__n);                          \
  })

#define __builtin_clz0uh(n)                     \
  ({                                            \
    ushort __n=(n);                             \
    __n |= __n >> 1;                            \
    __n |= __n >> 2;                            \
    __n |= __n >> 4;                            \
    __n |= __n >> 8;                            \
    16 - popcount(__n);                         \
  })

#define __builtin_clz0u(n)                      \
  ({                                            \
    uint __n=(n);                               \
    __n |= __n >> 1;                            \
    __n |= __n >> 2;                            \
    __n |= __n >> 4;                            \
    __n |= __n >> 8;                            \
    __n |= __n >> 16;                           \
    32 - popcount(__n);                         \
  })

#define __builtin_clz0ul(n)                     \
  ({                                            \
    ulong __n=(n);                              \
    __n |= __n >> 1;                            \
    __n |= __n >> 2;                            \
    __n |= __n >> 4;                            \
    __n |= __n >> 8;                            \
    __n |= __n >> 16;                           \
    __n |= __n >> 32;                           \
    64 - popcount(__n);                         \
  })

#define __builtin_clz0hh(n) __builtin_clz0uhh(n)
#define __builtin_clz0h(n)  __builtin_clz0uh(n)
#define __builtin_clz0(n)   __builtin_clz0u(n)
#define __builtin_clz0l(n)  __builtin_clz0ul(n)

#define clz0 clz
DEFINE_BUILTIN_G_G(clz0)
#undef clz0
