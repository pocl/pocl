/* OpenCL built-in library: builtin implementation templates for HSAIL

   Copyright (c) 2011-2013 Erik Schnetter
   Copyright (c) 2015 Michal Babej

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

/* TODO possibly prefix with HSAIL to not conflict with ../templates.h */

/**********************************************************************/

#define IMPLEMENT_BUILTIN_V_V(NAME, VTYPE, LO, HI)      \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a)                                         \
  {                                                     \
    return (VTYPE)(NAME(a.LO), NAME(a.HI));             \
  }

#define IMPLEMENT_BUILTIN_V_VV(NAME, VTYPE, LO, HI)     \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, VTYPE b)                                \
  {                                                     \
    return (VTYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI)); \
  }

#define IMPLEMENT_BUILTIN_V_VVV(NAME, VTYPE, LO, HI)                    \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, VTYPE b, VTYPE c)                                       \
  {                                                                     \
    return (VTYPE)(NAME(a.LO, b.LO, c.LO), NAME(a.HI, b.HI, c.HI));     \
  }

/**********************************************************************/

#define  IMPLEMENT_BUILTIN_TYPE_ALL_VECS(NAME, TYPE, STYPE)    \
  IMPLEMENT_BUILTIN_ ## TYPE(NAME, STYPE ## 2, lo, hi)          \
  IMPLEMENT_BUILTIN_ ## TYPE(NAME, STYPE ## 3, lo, s2)          \
  IMPLEMENT_BUILTIN_ ## TYPE(NAME, STYPE ## 4, lo, hi)          \
  IMPLEMENT_BUILTIN_ ## TYPE(NAME, STYPE ## 8, lo, hi)          \
  IMPLEMENT_BUILTIN_ ## TYPE(NAME, STYPE ## 16, lo, hi)

#define  IMPL_V_ALL(NAME, STYPE, BUILTIN, SUFFIX) \
  __attribute__((overloadable)) STYPE NAME(STYPE a) __asm("llvm."#BUILTIN#SUFFIX);   \
  IMPLEMENT_BUILTIN_TYPE_ALL_VECS(NAME, V_V, STYPE)

#define  IMPL_VV_ALL(NAME, STYPE, BUILTIN, SUFFIX) \
    __attribute__((overloadable)) STYPE NAME(STYPE a,STYPE b) __asm("llvm."#BUILTIN#SUFFIX);   \
  IMPLEMENT_BUILTIN_TYPE_ALL_VECS(NAME, V_VV, STYPE)

#define  IMPL_VVV_ALL(NAME, STYPE, BUILTIN, SUFFIX) \
    __attribute__((overloadable)) STYPE NAME(STYPE a, STYPE b,STYPE c) __asm("llvm."#BUILTIN#SUFFIX);   \
  IMPLEMENT_BUILTIN_TYPE_ALL_VECS(NAME, V_VVV, STYPE)

/**********************************************************************/

#define IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR) \
  {                                                                     \
    typedef VTYPE vtype;                                                \
    typedef STYPE stype;                                                \
    typedef VTYPE2 vtype2;                                              \
    typedef STYPE2 stype2;                                              \
    return EXPR;                                                        \
  }

#define IMPLEMENT_EXPR_V_V(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)    \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a)                                                         \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

#define IMPLEMENT_EXPR_V_VV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)   \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, VTYPE b)                                                \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

#define IMPLEMENT_EXPR_V_VVV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)  \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, VTYPE b, VTYPE c)                                       \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

#define IMPLEMENT_EXPR_V_VS(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)   \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, STYPE b)                                                \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

/**********************************************************************/

#define IMPLEMENT_CONV_V_V(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)    \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a)                                                         \
  {                                                                     \
    return convert_ ## VTYPE(NAME(convert_ ## VTYPE2(a)));                   \
  }

#define IMPLEMENT_CONV_V_VV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)    \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, VTYPE b)                                                \
  {                                                                     \
    return convert_ ## VTYPE(NAME(convert_ ## VTYPE2(a), convert_ ## VTYPE2(b))); \
  }

#define IMPLEMENT_CONV_V_VVV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)   \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, VTYPE b, VTYPE c)                                       \
  {                                                                     \
    return convert_ ## VTYPE(NAME(convert_ ## VTYPE2(a), convert_ ## VTYPE2(b),    \
                       convert_ ## VTYPE2(c)));                           \
  }

/**********************************************************************/

#define   IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, TYPE, IMPTYPE, STYPE, SSTYPE, EXPR)    \
  IMPLEMENT_ ## IMPTYPE ## TYPE(NAME, EXPR, STYPE, STYPE, SSTYPE, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## TYPE(NAME, EXPR, STYPE ## 2, STYPE, SSTYPE ## 2, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## TYPE(NAME, EXPR, STYPE ## 3, STYPE, SSTYPE ## 3, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## TYPE(NAME, EXPR, STYPE ## 4, STYPE, SSTYPE ## 4, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## TYPE(NAME, EXPR, STYPE ## 8, STYPE, SSTYPE ## 8, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## TYPE(NAME, EXPR, STYPE ## 16, STYPE, SSTYPE ## 16, SSTYPE)

/**********************************************************************/
#define EXPR_VV_ALL_SMALLINTS(NAME) \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VV, CONV_, char, int, "")          \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VV, CONV_, uchar, uint, "")        \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VV, CONV_, short, int, "")         \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VV, CONV_, ushort, uint, "")

#define EXPR_VVV_ALL_SMALLINTS(NAME)          \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, CONV_, char, int, "")          \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, CONV_, uchar, uint, "")        \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, CONV_, short, int, "")         \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, CONV_, ushort, uint, "")

/**********************************************************************/

#define DEFINE_BUILTIN_V_V_FP32(NAME, BUILTIN)          \
  IMPL_V_ALL(NAME, float, BUILTIN, .f32)

#define DEFINE_BUILTIN_V_V_FP64(NAME, BUILTIN)          \
  __IF_FP64(                                            \
  IMPL_V_ALL(NAME, double, BUILTIN, .f64))

#define DEFINE_BUILTIN_V_VV_FP32(NAME, BUILTIN)         \
  IMPL_VV_ALL(NAME, float, BUILTIN, .f32)

#define DEFINE_BUILTIN_V_VV_FP64(NAME, BUILTIN)         \
  __IF_FP64(                                            \
  IMPL_VV_ALL(NAME, double, BUILTIN, .f64))

/**********************************************************************/

#define DEFINE_EXPR_V_V_FP16_FP64(NAME, EXPR)                           \
  __IF_FP16(                                                            \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_V, EXPR_, half, short, EXPR))    \
  __IF_FP64(                                                            \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_V, EXPR_, double, long, EXPR))

/**********************************************************************/

#define DEFINE_BUILTIN_V_V_FP32_FP64(NAME, BUILTIN)                  \
  DEFINE_BUILTIN_V_V_FP32(NAME, BUILTIN)                                  \
  DEFINE_BUILTIN_V_V_FP64(NAME, BUILTIN)

#define DEFINE_BUILTIN_V_V_ONLY_FP32(NAME, BUILTIN, EXPR)          \
  DEFINE_BUILTIN_V_V_FP32(NAME, BUILTIN)                                  \
  DEFINE_EXPR_V_V_FP16_FP64(NAME, EXPR)

/**********************************************************************/

#define DEFINE_BUILTIN_V_VV_FP32_FP64(NAME, BUILTIN)                  \
  DEFINE_BUILTIN_V_VV_FP32(NAME, BUILTIN)                             \
  DEFINE_BUILTIN_V_VV_FP64(NAME, BUILTIN)

/*
#define DEFINE_BUILTIN_V_V_ONLY_FP32(NAME, BUILTIN, EXPR)             \
  DEFINE_BUILTIN_V_V_FP32(NAME, BUILTIN)                              \
  DEFINE_EXPR_V_V_FP16_FP64(NAME, EXPR)
*/

/**********************************************************************/

#define DEFINE_BUILTIN_V_VVV_FP32(NAME, BUILTIN)                      \
  IMPL_VVV_ALL(NAME, float, BUILTIN, .f32)


#define DEFINE_BUILTIN_V_VVV_FP64(NAME, BUILTIN)                      \
  __IF_FP64(                                                          \
  IMPL_VVV_ALL(NAME, double, BUILTIN, .f64))

/**********************************************************************/

#define DEFINE_BUILTIN_V_VVV_FP32_FP64(NAME, BUILTIN)                  \
  DEFINE_BUILTIN_V_VVV_FP32(NAME, BUILTIN)                             \
  DEFINE_BUILTIN_V_VVV_FP64(NAME, BUILTIN)

/**********************************************************************/

#define ASM_IMP(BUILTIN, SUFFIX, TYPE)

#define DEFINE_BUILTIN_V_VVV_S32(NAME, BUILTIN)                      \
  IMPL_VVV_ALL(NAME, int, BUILTIN, .s32)

#define DEFINE_BUILTIN_V_VVV_U32(NAME, BUILTIN)                      \
  IMPL_VVV_ALL(NAME, uint, BUILTIN, .u32)

#define DEFINE_BUILTIN_V_VVV_S64(NAME, BUILTIN)                      \
  IMPL_VVV_ALL(NAME, long, BUILTIN, .s64)

#define DEFINE_BUILTIN_V_VVV_U64(NAME, BUILTIN)                      \
  IMPL_VVV_ALL(NAME, ulong, BUILTIN, .u64)

#define DEFINE_BUILTIN_V_VVV_IS32(NAME, BUILTIN)                      \
  IMPL_VVV_ALL(NAME, int, BUILTIN, .i32)

#define DEFINE_BUILTIN_V_VVV_IU32(NAME, BUILTIN)                      \
  IMPL_VVV_ALL(NAME, uint, BUILTIN, .i32)

/**********************************************************************/

#define DEFINE_BUILTIN_V_VV_S32(NAME, BUILTIN)                      \
  IMPL_VV_ALL(NAME, int, BUILTIN, .s32)

#define DEFINE_BUILTIN_V_VV_U32(NAME, BUILTIN)                      \
  IMPL_VV_ALL(NAME, uint, BUILTIN, .u32)

#define DEFINE_BUILTIN_V_VV_S64(NAME, BUILTIN)                      \
  IMPL_VV_ALL(NAME, long, BUILTIN, .s64)

#define DEFINE_BUILTIN_V_VV_U64(NAME, BUILTIN)                      \
  IMPL_VV_ALL(NAME, ulong, BUILTIN, .u64)

#define DEFINE_BUILTIN_V_VV_IS32(NAME, BUILTIN)                      \
  IMPL_VV_ALL(NAME, int, BUILTIN, .i32)

#define DEFINE_BUILTIN_V_VV_IU32(NAME, BUILTIN)                      \
  IMPL_VV_ALL(NAME, uint, BUILTIN, .i32)

#define DEFINE_BUILTIN_V_VV_I64(NAME, BUILTIN)                      \
  IMPL_VV_ALL(NAME, long, BUILTIN, .i64)

/**********************************************************************/

#define DEFINE_BUILTIN_V_VV_SU_INT32_ONLY(NAME, SIGNED_BUILTIN, UNSIGNED_BUILTIN)   \
  DEFINE_BUILTIN_V_VV_IS32(NAME, SIGNED_BUILTIN)               \
  DEFINE_BUILTIN_V_VV_IU32(NAME, UNSIGNED_BUILTIN)

#define DEFINE_BUILTIN_V_VV_INT32_ONLY(NAME, BUILTIN)  \
  DEFINE_BUILTIN_V_VV_SU_INT32_ONLY(NAME, BUILTIN, BUILTIN)

#define DEFINE_BUILTIN_V_VVV_SU_INT32_ONLY(NAME, SIGNED_BUILTIN, UNSIGNED_BUILTIN)   \
  DEFINE_BUILTIN_V_VVV_IS32(NAME, SIGNED_BUILTIN)               \
  DEFINE_BUILTIN_V_VVV_IU32(NAME, UNSIGNED_BUILTIN)

#define DEFINE_BUILTIN_V_VVV_INT32_ONLY(NAME, BUILTIN)  \
  DEFINE_BUILTIN_V_VVV_SU_INT32_ONLY(NAME, BUILTIN, BUILTIN)

/**********************************************************************/

#define DEFINE_BUILTIN_V_VV_SU_ALL_INTS(NAME, SIGNED_BUILTIN, UNSIGNED_BUILTIN) \
  DEFINE_BUILTIN_V_VV_S32(NAME, SIGNED_BUILTIN)               \
  DEFINE_BUILTIN_V_VV_U32(NAME, UNSIGNED_BUILTIN)               \
  DEFINE_BUILTIN_V_VV_S64(NAME, SIGNED_BUILTIN)               \
  DEFINE_BUILTIN_V_VV_U64(NAME, UNSIGNED_BUILTIN)               \
  EXPR_VV_ALL_SMALLINTS(NAME)

#define DEFINE_BUILTIN_V_VV_II_ALL_INTS(NAME, SIGNED_BUILTIN, UNSIGNED_BUILTIN) \
  DEFINE_BUILTIN_V_VV_IS32(NAME, SIGNED_BUILTIN)               \
  DEFINE_BUILTIN_V_VV_I64(NAME, SIGNED_BUILTIN)               \
  EXPR_VV_ALL_SMALLINTS(NAME)

#define DEFINE_BUILTIN_V_VV_ALL_INTS(NAME, BUILTIN)    \
  DEFINE_BUILTIN_V_VV_SU_ALL_INTS(NAME, BUILTIN, BUILTIN)

#define DEFINE_BUILTIN_V_VVV_ALL_INTS(NAME, BUILTIN)   \
  DEFINE_BUILTIN_V_VVV_S32(NAME, BUILTIN)              \
  DEFINE_BUILTIN_V_VVV_U32(NAME, BUILTIN)              \
  DEFINE_BUILTIN_V_VVV_S64(NAME, BUILTIN)              \
  DEFINE_BUILTIN_V_VVV_U64(NAME, BUILTIN)              \
  EXPR_VVV_ALL_SMALLINTS(NAME)

#define DEFINE_EXPR_V_VVV_ALL_INTS(NAME, EXPR)         \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, EXPR_, int, int, EXPR)      \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, EXPR_, uint, uint, EXPR)    \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, EXPR_, long, long, EXPR)    \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VVV, EXPR_, ulong, ulong, EXPR)  \
  EXPR_VVV_ALL_SMALLINTS(NAME)

/**********************************************************************/

#define DEFINE_EXPR_V_VS_FP32_FP64(NAME, EXPR)           \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VS, EXPR_, float, float, EXPR)   \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VS, EXPR_, double, double, EXPR) \

#define DEFINE_EXPR_V_VPV_FP32_FP64(NAME, EXPR)       \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VPV, EXPR_, float, float, EXPR)   \
  IMPLEMENT_EXPR_TYPE_ALL_VECS(NAME, V_VPV, EXPR_, double, double, EXPR) \
