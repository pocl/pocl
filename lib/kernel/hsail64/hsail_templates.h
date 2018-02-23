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

// Make vectorized versions of a scalar builtin using Divide-n-Conquer

#define IMPLEMENT_VECWITHSCALARS_V_V(NAME, VTYPE, ITYPE, IVTYPE, LO, HI)      \
  VTYPE _CL_OVERLOADABLE                  \
  NAME(VTYPE a)                                         \
  {                                                     \
    return (VTYPE)(NAME(a.LO), NAME(a.HI));             \
  }

#define IMPLEMENT_VECWITHSCALARS_V_VV(NAME, VTYPE, ITYPE, IVTYPE, LO, HI)     \
  VTYPE _CL_OVERLOADABLE                  \
  NAME(VTYPE a, VTYPE b)                                \
  {                                                     \
    return (VTYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI)); \
  }

#define IMPLEMENT_VECWITHSCALARS_V_VVV(NAME, VTYPE, ITYPE, IVTYPE, LO, HI)    \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a, VTYPE b, VTYPE c)                                       \
  {                                                                     \
    return (VTYPE)(NAME(a.LO, b.LO, c.LO), NAME(a.HI, b.HI, c.HI));     \
  }

// ldexp
#define IMPLEMENT_VECWITHSCALARS_V_VI(NAME, VTYPE, ITYPE, IVTYPE, LO, HI)     \
  VTYPE _CL_OVERLOADABLE                  \
  NAME(VTYPE a, IVTYPE b)                               \
  {                                                     \
    return (VTYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI)); \
  }                                                     \
  VTYPE _CL_OVERLOADABLE                  \
  NAME(VTYPE a, ITYPE b)                                \
  {                                                     \
    return (VTYPE)(NAME(a, (IVTYPE)b)); \
  }

// ilogb
#define IMPLEMENT_VECWITHSCALARS_I_V(NAME, VTYPE, ITYPE, IVTYPE, LO, HI)     \
  IVTYPE _CL_OVERLOADABLE                  \
  NAME(VTYPE a)                               \
  {                                                     \
    return (IVTYPE)(NAME(a.LO), NAME(a.HI)); \
  }

/**********************************************************************/

#define  IMPLEMENT_VECWITHSCALARS(NAME, TYPE, STYPE, ITYPE)    \
  IMPLEMENT_VECWITHSCALARS_ ## TYPE(NAME, STYPE ## 2, ITYPE, ITYPE ## 2, lo, hi)          \
  IMPLEMENT_VECWITHSCALARS_ ## TYPE(NAME, STYPE ## 3, ITYPE, ITYPE ## 3, lo, s2)          \
  IMPLEMENT_VECWITHSCALARS_ ## TYPE(NAME, STYPE ## 4, ITYPE, ITYPE ## 4, lo, hi)          \
  IMPLEMENT_VECWITHSCALARS_ ## TYPE(NAME, STYPE ## 8, ITYPE, ITYPE ## 8, lo, hi)          \
  IMPLEMENT_VECWITHSCALARS_ ## TYPE(NAME, STYPE ## 16, ITYPE, ITYPE ## 16, lo, hi)

/**********************************************************************/

// ASM for scalar, DnC for vectors
#define  IMPLEMENT_LLVM_INTRIN_V_V_ALL(NAME, STYPE, BUILTIN, SUFFIX) \
  STYPE NAME ## _internal_v_ ## STYPE(STYPE a)  __asm("llvm."#BUILTIN#SUFFIX);  \
  _CL_OVERLOADABLE STYPE NAME(STYPE a) { return NAME ## _internal_v_ ## STYPE(a); } \
  IMPLEMENT_VECWITHSCALARS(NAME, V_V, STYPE, STYPE)

#define  IMPLEMENT_LLVM_INTRIN_V_VV_ALL(NAME, STYPE, BUILTIN, SUFFIX) \
  STYPE NAME ## _internal_vv_ ## STYPE(STYPE a, STYPE b)  __asm("llvm."#BUILTIN#SUFFIX);  \
  _CL_OVERLOADABLE STYPE NAME(STYPE a, STYPE b) { return NAME ## _internal_vv_ ## STYPE(a, b); } \
  IMPLEMENT_VECWITHSCALARS(NAME, V_VV, STYPE, STYPE)

#define  IMPLEMENT_LLVM_INTRIN_V_VVV_ALL(NAME, STYPE, BUILTIN, SUFFIX) \
  STYPE NAME ## _internal_vvv_ ## STYPE(STYPE a, STYPE b,STYPE c)  __asm("llvm."#BUILTIN#SUFFIX);  \
  _CL_OVERLOADABLE STYPE NAME(STYPE a, STYPE b,STYPE c) { return NAME ## _internal_vvv_ ## STYPE(a, b, c); } \
  IMPLEMENT_VECWITHSCALARS(NAME, V_VVV, STYPE, STYPE)

#define  IMPLEMENT_LLVM_INTRIN_V_VI_ALL(NAME, STYPE, ITYPE, BUILTIN, SUFFIX) \
  STYPE NAME ## _internal_vi_ ## STYPE(STYPE a, ITYPE b) __asm("llvm."#BUILTIN#SUFFIX);   \
  _CL_OVERLOADABLE STYPE NAME(STYPE a, ITYPE b) { return NAME ## _internal_vi_ ## STYPE(a, b); } \
  IMPLEMENT_VECWITHSCALARS(NAME, V_VI, STYPE, ITYPE)

/**********************************************************************/

// Implement a builtin using an Expression

#define IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR) \
  {                                                                     \
    typedef VTYPE vtype;                                                \
    typedef STYPE stype;                                                \
    typedef VTYPE2 vtype2;                                              \
    typedef STYPE2 stype2;                                              \
    return EXPR;                                                        \
  }

#define IMPLEMENT_EXPR_V_V(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)    \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a)                                                         \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

#define IMPLEMENT_EXPR_V_VV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)   \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a, VTYPE b)                                                \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

#define IMPLEMENT_EXPR_V_VVV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)  \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a, VTYPE b, VTYPE c)                                       \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

#define IMPLEMENT_EXPR_V_VS(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)   \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a, STYPE b)                                                \
  IMPL_BODY(VTYPE, STYPE, VTYPE2, STYPE2, EXPR)

/**********************************************************************/

// Converts, useful for when you only have builtin for uint32 & uint64,
// but need opencl for all the integers
#define IMPLEMENT_CONV_V_V(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)    \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a)                                                         \
  {                                                                     \
    return convert_ ## VTYPE(NAME(convert_ ## VTYPE2(a)));                   \
  }

#define IMPLEMENT_CONV_V_VV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)    \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a, VTYPE b)                                                \
  {                                                                     \
    return convert_ ## VTYPE(NAME(convert_ ## VTYPE2(a), convert_ ## VTYPE2(b))); \
  }

#define IMPLEMENT_CONV_V_VVV(NAME, EXPR, VTYPE, STYPE, VTYPE2, STYPE2)   \
  VTYPE _CL_OVERLOADABLE                                  \
  NAME(VTYPE a, VTYPE b, VTYPE c)                                       \
  {                                                                     \
    return convert_ ## VTYPE(NAME(convert_ ## VTYPE2(a), convert_ ## VTYPE2(b),    \
                       convert_ ## VTYPE2(c)));                           \
  }
/**********************************************************************/

#define IMPLEMENT_EXPR_VECS_ONLY(NAME, ARGTYPE, IMPTYPE, STYPE, SSTYPE, EXPR)    \
  IMPLEMENT_ ## IMPTYPE ## ARGTYPE(NAME, EXPR, STYPE ## 2, STYPE, SSTYPE ## 2, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## ARGTYPE(NAME, EXPR, STYPE ## 3, STYPE, SSTYPE ## 3, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## ARGTYPE(NAME, EXPR, STYPE ## 4, STYPE, SSTYPE ## 4, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## ARGTYPE(NAME, EXPR, STYPE ## 8, STYPE, SSTYPE ## 8, SSTYPE)          \
  IMPLEMENT_ ## IMPTYPE ## ARGTYPE(NAME, EXPR, STYPE ## 16, STYPE, SSTYPE ## 16, SSTYPE)

#define IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, TYPE, IMPTYPE, STYPE, SSTYPE, EXPR)    \
  IMPLEMENT_ ## IMPTYPE ## TYPE(NAME, EXPR, STYPE, STYPE, SSTYPE, SSTYPE)          \
  IMPLEMENT_EXPR_VECS_ONLY(NAME, TYPE, IMPTYPE, STYPE, SSTYPE, EXPR)

/**********************************************************************/

// EXPR for scalar, DnC for vectors, useful for GCC builtins
#define IMPLEMENT_EXPR_ALL(NAME, ARGTYPE, EXPR32, EXPR64)                      \
  IMPLEMENT_EXPR_ ## ARGTYPE(NAME, EXPR32, float, float, float, float)         \
  IMPLEMENT_VECWITHSCALARS(NAME, ARGTYPE, float, int)                   \
  IMPLEMENT_EXPR_ ## ARGTYPE(NAME, EXPR64, double, double, double, double)     \
  IMPLEMENT_VECWITHSCALARS(NAME, ARGTYPE, double, long)

#define IMPLEMENT_EXPR_ALL_SINGLE(NAME, ARGTYPE, EXPR)                       \
  IMPLEMENT_EXPR_ ## ARGTYPE(NAME, EXPR, float, float, float, float)         \
  IMPLEMENT_VECWITHSCALARS(NAME, ARGTYPE, float, int)                 \
  IMPLEMENT_EXPR_ ## ARGTYPE(NAME, EXPR, double, double, double, double)     \
  IMPLEMENT_VECWITHSCALARS(NAME, ARGTYPE, double, long)

/**********************************************************************/

// Convert from char/shorts to ints

#define EXPR_V_VV_ALL_SMALLINTS(NAME) \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VV, CONV_, char, int, "")          \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VV, CONV_, uchar, uint, "")        \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VV, CONV_, short, int, "")         \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VV, CONV_, ushort, uint, "")

#define EXPR_V_VVV_ALL_SMALLINTS(NAME)          \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, CONV_, char, int, "")          \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, CONV_, uchar, uint, "")        \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, CONV_, short, int, "")         \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, CONV_, ushort, uint, "")

/**********************************************************************/

// Define OpenCL runtime func via LLVM intrinsics (IMPL_*_ALL)

#define DEFINE_LLVM_INTRIN_FP32_FP64(NAME, ARGTYPE, BUILTIN, EXPR16)       \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, float, BUILTIN, .f32)                     \
  __IF_FP64(                                                               \
    IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, double, BUILTIN, .f64))                 \
  __IF_FP16(                                                               \
    IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, ARGTYPE, EXPR_, half, short, EXPR16))

#define DEFINE_LLVM_INTRIN_ONLY_FP32(NAME, ARGTYPE, BUILTIN, EXPR64, EXPR16)    \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, float, BUILTIN, .f32)                          \
  __IF_FP64(                                                                    \
    IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, ARGTYPE, EXPR_, double, long, EXPR64)) \
  __IF_FP16(                                                                    \
    IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, ARGTYPE, EXPR_, half, short, EXPR16))  \

/**********************************************************************/

// ldexp, doesn't work yet

#define DEFINE_LLVM_INTRIN_V_VI_FP32_FP64(NAME, BUILTIN)                \
  IMPLEMENT_LLVM_INTRIN_V_VI_ALL(NAME, float, int, BUILTIN, .f32)                          \
  __IF_FP16(                                                            \
    IMPLEMENT_LLVM_INTRIN_V_VI_ALL(NAME, half, int, BUILTIN, .f32))                        \
  __IF_FP64(                                                            \
    IMPLEMENT_LLVM_INTRIN_V_VI_ALL(NAME, double, int, BUILTIN, .f64))                      \

/**********************************************************************/

// For mul_hi  /* - has hsail.smulhi.i32 & hsail.umulhi.i32 */
#define DEFINE_LLVM_INTRIN_SU_INT32_ONLY(NAME, ARGTYPE, SIGNED_BUILTIN, UNSIGNED_BUILTIN)   \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, int, SIGNED_BUILTIN, .i32)                         \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, uint, UNSIGNED_BUILTIN, .i32)                       \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, long, SIGNED_BUILTIN, .i64)                         \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, ulong, UNSIGNED_BUILTIN, .i64)                       \
  EXPR_ ## ARGTYPE ## _ALL_SMALLINTS(NAME)
/*
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, short, SIGNED_BUILTIN, .i16)                         \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, ushort, UNSIGNED_BUILTIN, .i16)                       \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, char, SIGNED_BUILTIN, .i8)                         \
  IMPLEMENT_LLVM_INTRIN_ ## ARGTYPE ## _ALL(NAME, uchar, UNSIGNED_BUILTIN, .i8)
*/


// For mad_hi, defined as mul_hi(a,b)+c
#define DEFINE_EXPR_V_VVV_ALL_INTS(NAME, EXPR)         \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, int, int, EXPR)      \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, uint, uint, EXPR)    \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, long, long, EXPR)    \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, ulong, ulong, EXPR)  \
  EXPR_V_VVV_ALL_SMALLINTS(NAME)
/*
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, long, long, EXPR)    \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, ulong, ulong, EXPR)  \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, long, long, EXPR)    \
  IMPLEMENT_EXPR_VECS_AND_SCALAR(NAME, V_VVV, EXPR_, ulong, ulong, EXPR)  \
*/

/**********************************************************************/

// Vector - Pointer to Vector, for frexp() / fract()

#define VPV_BODY(N, NAME, VTYPE, STYPE, PTYPE, ADDRSP)            \
{                                                                     \
  typedef VTYPE vtype;                                                \
  typedef STYPE stype;                                                \
  typedef PTYPE ptype;                                                \
  ADDRSP ptype* temp = (ADDRSP ptype*)b;                              \
  typedef union {                                                             \
    stype s[N];                                   \
    vtype v;                                       \
  } E;                                             \
  E e; e.v = a;                                     \
  E out;                                                          \
  for (unsigned i=0; i<N; i++)                                        \
    out.s[i] = NAME(e.s[i], temp+i);                               \
  return out.v;                                                         \
}                                                                     \

#define IMPLEMENT_EXPR_V_VP_N(NAME, N, STYPE, PTYPE)  \
  STYPE ## N _CL_OVERLOADABLE                                  \
  NAME(STYPE ## N a, __global PTYPE ## N *b)                                      \
  VPV_BODY(N, NAME, STYPE ## N, STYPE, PTYPE, __global)                              \
  STYPE ## N _CL_OVERLOADABLE                                  \
  NAME(STYPE ## N a, __local PTYPE ## N *b)                                       \
  VPV_BODY(N, NAME, STYPE ## N, STYPE, PTYPE, __local)                               \
  STYPE ## N _CL_OVERLOADABLE                                  \
  NAME(STYPE ## N a, __private PTYPE ## N *b)                                     \
  VPV_BODY(N, NAME, STYPE ## N, STYPE, PTYPE, __private)                             \

#define IMPLEMENT_EXPR_V_VP_ALL(NAME, STYPE, PTYPE)                \
  IMPLEMENT_EXPR_V_VP_N(NAME, 2, STYPE, PTYPE)                     \
  IMPLEMENT_EXPR_V_VP_N(NAME, 3, STYPE, PTYPE)                     \
  IMPLEMENT_EXPR_V_VP_N(NAME, 4, STYPE, PTYPE)                     \
  IMPLEMENT_EXPR_V_VP_N(NAME, 8, STYPE, PTYPE)                     \
  IMPLEMENT_EXPR_V_VP_N(NAME, 16, STYPE, PTYPE)
