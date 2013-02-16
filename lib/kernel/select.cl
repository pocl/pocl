/* OpenCL built-in library: select()

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
   FITNESS FOR A PARTICULAR PURPOSE AND NONORDEREDRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "templates.h"

/* select needs to be implemented slightly differently between scalar,
   signed vector, and unsigned vector types, due to how the relational
   operators differ between them and signed/unsigned arithmetic.

   TODO: clang appears to be generating rather inefficient code for the
   vector versions, it's extracting each component and performing
   piece-wise comparisons instead of emitting the llvm ir "select"
   instruction. It might be worth investigating fixing this.
*/

#define IMPLEMENT_SELECT_SCALAR(GTYPE) \
  GTYPE __attribute__ ((overloadable)) \
  select(GTYPE a, GTYPE b, GTYPE c)    \
  {                                    \
     return c ? b : a;                 \
  }

IMPLEMENT_SELECT_SCALAR(char)
IMPLEMENT_SELECT_SCALAR(uchar)
IMPLEMENT_SELECT_SCALAR(short)
IMPLEMENT_SELECT_SCALAR(ushort)
IMPLEMENT_SELECT_SCALAR(int)
IMPLEMENT_SELECT_SCALAR(uint)
__IF_INT64(
IMPLEMENT_SELECT_SCALAR(long)
IMPLEMENT_SELECT_SCALAR(ulong))

#define IMPLEMENT_SELECT_VECTOR_SIGNED(GTYPE) \
  GTYPE __attribute__ ((overloadable))        \
  select(GTYPE a, GTYPE b, GTYPE c)           \
  {                                           \
     return (c < (GTYPE)0) ? b : a;           \
  }

IMPLEMENT_SELECT_VECTOR_SIGNED(char2)
IMPLEMENT_SELECT_VECTOR_SIGNED(char3)
IMPLEMENT_SELECT_VECTOR_SIGNED(char4)
IMPLEMENT_SELECT_VECTOR_SIGNED(char8)
IMPLEMENT_SELECT_VECTOR_SIGNED(char16)
IMPLEMENT_SELECT_VECTOR_SIGNED(short2)
IMPLEMENT_SELECT_VECTOR_SIGNED(short3)
IMPLEMENT_SELECT_VECTOR_SIGNED(short4)
IMPLEMENT_SELECT_VECTOR_SIGNED(short8)
IMPLEMENT_SELECT_VECTOR_SIGNED(short16)
IMPLEMENT_SELECT_VECTOR_SIGNED(int2)
IMPLEMENT_SELECT_VECTOR_SIGNED(int3)
IMPLEMENT_SELECT_VECTOR_SIGNED(int4)
IMPLEMENT_SELECT_VECTOR_SIGNED(int8)
IMPLEMENT_SELECT_VECTOR_SIGNED(int16)
__IF_INT64(
IMPLEMENT_SELECT_VECTOR_SIGNED(long2)
IMPLEMENT_SELECT_VECTOR_SIGNED(long3)
IMPLEMENT_SELECT_VECTOR_SIGNED(long4)
IMPLEMENT_SELECT_VECTOR_SIGNED(long8)
IMPLEMENT_SELECT_VECTOR_SIGNED(long16))

#define IMPLEMENT_SELECT_VECTOR_UNSIGNED(UGTYPE, IGTYPE) \
  UGTYPE __attribute__ ((overloadable))                  \
  select(UGTYPE a, UGTYPE b, UGTYPE c)                   \
  {                                                      \
     return (as_##IGTYPE(c) < (IGTYPE)0) ? b : a;        \
  }

IMPLEMENT_SELECT_VECTOR_UNSIGNED(uchar2, char2)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uchar3, char3)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uchar4, char4)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uchar8, char8)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uchar16, char16)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ushort2, short2)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ushort3, short3)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ushort4, short4)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ushort8, short8)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ushort16, short16)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uint2, int2)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uint3, int3)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uint4, int4)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uint8, int8)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(uint16, int16)
__IF_INT64(
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ulong2, long2)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ulong3, long3)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ulong4, long4)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ulong8, long8)
IMPLEMENT_SELECT_VECTOR_UNSIGNED(ulong16, long16))

#if (__clang_major__ > 3) || ((__clang_major__ == 3) && (__clang_minor__ > 3))
DEFINE_EXPR_V_VVJ(select, c ? b : a)
#else
/* This segfaults Clang 3.0, so we work around. */
DEFINE_EXPR_V_VVJ(select,
                  ({
                    jtype result = c ? *(jtype*)&b : *(jtype*)&a;
                    *(vtype*)&result;
                  }))
#endif

