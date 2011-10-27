/* OpenCL built-in library: clz()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#include "templates.h"

// Intel: LZCNT (and POPCNT)

#define __builtin_clzhh  __builtin_clz
#define __builtin_clzh   __builtin_clz
#define __builtin_clzuhh __builtin_clz
#define __builtin_clzuh  __builtin_clz
#define __builtin_clzu   __builtin_clz
#define __builtin_clzul  __builtin_clzl

DEFINE_BUILTIN_G_G(clz)

#if 0

/* Count ones */
#define CO(b)                                                           \
  ({                                                                    \
    ugtype c = b;                                                       \
    int bitmask = CHAR_BIT * sizeof(sugtype) - 1;                       \
    c -= ((c >> (sugtype)1) & (ugtype)0x5555555555555555UL);            \
    c = (((c >> (sugtype)2) & (ugtype)0x3333333333333333UL) +           \
         (c & (ugtype)0x3333333333333333UL));                           \
    c = (((c >> (sugtype)4) + c) & (ugtype)0x0f0f0f0f0f0f0f0fUL);       \
    c += (c >> (sugtype)( 8 & bitmask));                                \
    c += (c >> (sugtype)(16 & bitmask));                                \
    c += (c >> (sugtype)(32 & bitmask));                                \
    c & (ugtype)0xff;                                                   \
  })

/* Count leading zeros */
#define CLZ(a)                                          \
  ({                                                    \
    ugtype b = a;                                       \
    sugtype bits = CHAR_BIT * sizeof(sugtype);          \
    int bitmask = CHAR_BIT * sizeof(sugtype) - 1;       \
    b |= (b >> (sugtype)1);                             \
    b |= (b >> (sugtype)2);                             \
    b |= (b >> (sugtype)4);                             \
    b |= (b >> (sugtype)( 8 & bitmask));                \
    b |= (b >> (sugtype)(16 & bitmask));                \
    b |= (b >> (sugtype)(32 & bitmask));                \
    (ugtype)bits - CO(b);                               \
  })

DEFINE_EXPR_G_G(clz,
                ({
                  ugtype lz = CLZ(*(ugtype*)&a);
                  *(gtype*)&lz;
                }))

#endif
