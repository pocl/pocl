/* OpenCL built-in library: convert_type()

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

/* SRC and DST must be scalars */
#define DEFINE_CONVERT_TYPE(SRC, DST)           \
  DST _cl_overloadable convert_##DST(SRC a)     \
  {                                             \
    return (DST)a;                              \
  }

/* implementing vector SRC and DST in terms of scalars */
#define DEFINE_CONVERT_TYPE_HALF(SRC, DST, HALFDST)                     \
  DST _cl_overloadable convert_##DST(SRC a)                             \
  {                                                                     \
    return (DST)(convert_##HALFDST(a.lo), convert_##HALFDST(a.hi));     \
  }

#define DEFINE_CONVERT_TYPE_012(SRC, DST, DST01, DST2)          \
  DST _cl_overloadable convert_##DST(SRC a)                     \
  {                                                             \
    return (DST)(convert_##DST01(a.s01), convert_##DST2(a.s2)); \
  }

/* SRC and DST may be vectors */
#define DEFINE_CONVERT_TYPE_SAT(SRC, DST, SIZE)                         \
  DST##SIZE _cl_overloadable                                            \
    convert_##DST##SIZE##_sat(SRC##SIZE a)                              \
  {                                                                     \
    int const src_float    = (SRC)0.1f > (SRC)0;                        \
    int const src_unsigned = -(SRC)1 > (SRC)0;                          \
    int const dst_unsigned = -(DST)1 > (DST)0;                          \
    int const src_size = sizeof(SRC);                                   \
    int const dst_size = sizeof(DST);                                   \
    if (src_float) {                                                    \
      if (dst_unsigned) {                                               \
        DST const DST_MAX = (DST)0 - (DST)1;                            \
        return (convert_##DST##SIZE(a > (SRC)DST_MAX) ? (DST##SIZE)DST_MAX : \
                convert_##DST##SIZE(a));                                \
      } else { /* dst is signed */                                      \
        DST const DST_MIN = (DST)1 << (DST)(CHAR_BIT * dst_size - 1);   \
        DST const DST_MAX = DST_MIN - (DST)1;                           \
        return (convert_##DST##SIZE(a < (SRC)DST_MIN) ? (DST##SIZE)DST_MIN : \
                convert_##DST##SIZE(a > (SRC)DST_MAX) ? (DST##SIZE)DST_MAX : \
                convert_##DST##SIZE(a));                                \
      }                                                                 \
    } else if (src_unsigned) {                                          \
      if (dst_unsigned) {                                               \
        if (dst_size >= src_size) return convert_##DST##SIZE(a);        \
        DST const DST_MAX = (DST)0 - (DST)1;                            \
        return (convert_##DST##SIZE(a > (SRC)DST_MAX) ? (DST##SIZE)DST_MAX : \
                convert_##DST##SIZE(a));                                \
      } else { /* dst is signed */                                      \
        if (dst_size > src_size) return convert_##DST##SIZE(a);         \
        DST const DST_MAX = (DST)1 << (DST)(CHAR_BIT * dst_size);       \
        return (convert_##DST##SIZE(a > (SRC)DST_MAX) ? (DST##SIZE)DST_MAX : \
                convert_##DST##SIZE(a));                                \
      }                                                                 \
    } else { /* src is signed */                                        \
      if (dst_unsigned) {                                               \
        if (dst_size >= src_size) {                                     \
          return (convert_##DST##SIZE(a < (SRC)0) ? (DST##SIZE)0 :      \
                  convert_##DST##SIZE(a));                              \
        }                                                               \
        DST const DST_MAX = (DST)0 - (DST)1;                            \
        return (convert_##DST##SIZE(a < (SRC)0      ) ? (DST##SIZE)0 :  \
                convert_##DST##SIZE(a > (SRC)DST_MAX) ? (DST##SIZE)DST_MAX : \
                convert_##DST##SIZE(a));                                \
      } else { /* dst is signed */                                      \
        if (dst_size >= src_size) return convert_##DST##SIZE(a);        \
        DST const DST_MIN = (DST)1 << (DST)(CHAR_BIT * dst_size - 1);   \
        DST const DST_MAX = DST_MIN - (DST)1;                           \
        return (convert_##DST##SIZE(a < (SRC)DST_MIN) ? (DST##SIZE)DST_MIN : \
                convert_##DST##SIZE(a > (SRC)DST_MAX) ? (DST##SIZE)DST_MAX : \
                convert_##DST##SIZE(a));                                \
      }                                                                 \
    }                                                                   \
  }



#define DEFINE_CONVERT_TYPE_ALL(SRC, DST)                       \
  DEFINE_CONVERT_TYPE     (SRC    , DST    )                    \
  DEFINE_CONVERT_TYPE_HALF(SRC##2 , DST##2 , DST)               \
  DEFINE_CONVERT_TYPE_012 (SRC##3 , DST##3 , DST##2, DST)       \
  DEFINE_CONVERT_TYPE_HALF(SRC##4 , DST##4 , DST##2)            \
  DEFINE_CONVERT_TYPE_HALF(SRC##8 , DST##8 , DST##4)            \
  DEFINE_CONVERT_TYPE_HALF(SRC##16, DST##16, DST##8)

#define DEFINE_CONVERT_TYPE_SAT_ALL(SRC, DST)   \
  DEFINE_CONVERT_TYPE_SAT(SRC, DST,   )         \
  DEFINE_CONVERT_TYPE_SAT(SRC, DST,  2)         \
  DEFINE_CONVERT_TYPE_SAT(SRC, DST,  3)         \
  DEFINE_CONVERT_TYPE_SAT(SRC, DST,  4)         \
  DEFINE_CONVERT_TYPE_SAT(SRC, DST,  8)         \
  DEFINE_CONVERT_TYPE_SAT(SRC, DST, 16)



#define DEFINE_CONVERT_TYPE_ALL_DST(SRC)        \
  DEFINE_CONVERT_TYPE_ALL    (SRC, uchar )      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, uchar )      \
  DEFINE_CONVERT_TYPE_ALL    (SRC, char  )      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, char  )      \
  DEFINE_CONVERT_TYPE_ALL    (SRC, ushort)      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, ushort)      \
  DEFINE_CONVERT_TYPE_ALL    (SRC, short )      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, short )      \
  DEFINE_CONVERT_TYPE_ALL    (SRC, uint  )      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, uint  )      \
  DEFINE_CONVERT_TYPE_ALL    (SRC, int   )      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, int   )      \
  __IF_INT64(                                   \
  DEFINE_CONVERT_TYPE_ALL    (SRC, ulong )      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, ulong )      \
  DEFINE_CONVERT_TYPE_ALL    (SRC, long  )      \
  DEFINE_CONVERT_TYPE_SAT_ALL(SRC, long  ))     \
  DEFINE_CONVERT_TYPE_ALL    (SRC, float )      \
  __IF_FP64(                                    \
  DEFINE_CONVERT_TYPE_ALL    (SRC, double))

DEFINE_CONVERT_TYPE_ALL_DST(uchar )
DEFINE_CONVERT_TYPE_ALL_DST(char  )
DEFINE_CONVERT_TYPE_ALL_DST(ushort)
DEFINE_CONVERT_TYPE_ALL_DST(short )
DEFINE_CONVERT_TYPE_ALL_DST(uint  )
DEFINE_CONVERT_TYPE_ALL_DST(int   )
__IF_INT64(
DEFINE_CONVERT_TYPE_ALL_DST(ulong )
DEFINE_CONVERT_TYPE_ALL_DST(long  ))
DEFINE_CONVERT_TYPE_ALL_DST(float )
__IF_FP64(
DEFINE_CONVERT_TYPE_ALL_DST(double))
