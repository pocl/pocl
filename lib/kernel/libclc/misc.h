/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#define SNAN 0x001
#define QNAN 0x002
#define NINF 0x004
#define NNOR 0x008
#define NSUB 0x010
#define NZER 0x020
#define PZER 0x040
#define PSUB 0x080
#define PNOR 0x100
#define PINF 0x200

#define HAVE_BITALIGN() (0)

#define MATH_DIVIDE(X, Y) ((X) / (Y))
#define MATH_RECIP(X) (1.0f / (X))
#define MATH_SQRT(X) sqrt(X)

#define SIGNBIT_SP32      0x80000000
#define EXSIGNBIT_SP32    0x7fffffff
#define EXPBITS_SP32      0x7f800000
#define MANTBITS_SP32     0x007fffff
#define MANTSIGNBITS_SP32 0x807fffff
#define ONEEXPBITS_SP32   0x3f800000
#define TWOEXPBITS_SP32   0x40000000
#define HALFEXPBITS_SP32  0x3f000000
#define IMPBIT_SP32       0x00800000
#define QNANBITPATT_SP32  0x7fc00000
#define INDEFBITPATT_SP32 0xffc00000
#define PINFBITPATT_SP32  0x7f800000
#define NINFBITPATT_SP32  0xff800000
#define EXPBIAS_SP32      127
#define EXPSHIFTBITS_SP32 23
#define BIASEDEMIN_SP32   1
#define EMIN_SP32         -126
#define BIASEDEMAX_SP32   254
#define EMAX_SP32         127
#define LAMBDA_SP32       1.0e30
#define MANTLENGTH_SP32   24
#define BASEDIGITS_SP32   7
#define ISNEG_SP32(x)     (as_itype(x) & (itype)SIGNBIT_SP32)
#define vINFINITY_SP32    (as_vtype((utype)PINFBITPATT_SP32))
#define vNINFINITY_SP32   (as_vtype((utype)NINFBITPATT_SP32))
#define vNAN_SP32         (as_vtype((utype)QNANBITPATT_SP32))
#define vZERO_SP32        (vtype)0.0f
#define vONE_SP32        (vtype)1.0f

#ifdef cl_khr_fp64

#define SIGNBIT_DP64      0x8000000000000000L
#define EXSIGNBIT_DP64    0x7fffffffffffffffL
#define EXPBITS_DP64      0x7ff0000000000000L
#define MANTBITS_DP64     0x000fffffffffffffL
#define MANTSIGNBITS_DP64 0x800fffffffffffffL
#define ONEEXPBITS_DP64   0x3ff0000000000000L
#define TWOEXPBITS_DP64   0x4000000000000000L
#define HALFEXPBITS_DP64  0x3fe0000000000000L
#define IMPBIT_DP64       0x0010000000000000L
#define QNANBITPATT_DP64  0x7ff8000000000000L
#define INDEFBITPATT_DP64 0xfff8000000000000L
#define PINFBITPATT_DP64  0x7ff0000000000000L
#define NINFBITPATT_DP64  0xfff0000000000000L
#define EXPBIAS_DP64      1023
#define EXPSHIFTBITS_DP64 52
#define BIASEDEMIN_DP64   1
#define EMIN_DP64         -1022
#define BIASEDEMAX_DP64   2046 /* 0x7fe */
#define EMAX_DP64         1023 /* 0x3ff */
#define LAMBDA_DP64       1.0e300
#define MANTLENGTH_DP64   53
#define BASEDIGITS_DP64   15
#define ISNEG_DP64(x)     (as_itype(x) & (itype)SIGNBIT_DP64)
#define vINFINITY_DP64    (as_vtype((utype)PINFBITPATT_DP64))
#define vNINFINITY_DP64   (as_vtype((utype)NINFBITPATT_DP64))
#define vNAN_DP64         (as_vtype((utype)QNANBITPATT_DP64))
#define vZERO_DP64        (vtype)0.0
#define vONE_DP64         (vtype)1.0

#endif // cl_khr_fp64

#define ALIGNED(x)  __attribute__((aligned(x)))


#ifdef cl_khr_fp64

typedef struct { double lo,hi; } v2double;
typedef struct { double2 lo,hi; } v2double2;
typedef struct { double3 lo,hi; } v2double3;
typedef struct { double4 lo,hi; } v2double4;
typedef struct { double8 lo,hi; } v2double8;
typedef struct { double16 lo,hi; } v2double16;

#endif

typedef struct { float lo,hi; } v2float;
typedef struct { float2 lo,hi; } v2float2;
typedef struct { float3 lo,hi; } v2float3;
typedef struct { float4 lo,hi; } v2float4;
typedef struct { float8 lo,hi; } v2float8;
typedef struct { float16 lo,hi; } v2float16;

// for PI tables sin / cos
typedef struct { uint s0, s1, s2, s3; } v4uint;
typedef struct { uint2 s0, s1, s2, s3; } v4uint2;
typedef struct { uint3 s0, s1, s2, s3; } v4uint3;
typedef struct { uint4 s0, s1, s2, s3; } v4uint4;
typedef struct { uint8 s0, s1, s2, s3; } v4uint8;
typedef struct { uint16 s0, s1, s2, s3; } v4uint16;

// for PI tables sin / cos
typedef struct { int s0, s1, s2, s3; } v4int;
typedef struct { int2 s0, s1, s2, s3; } v4int2;
typedef struct { int3 s0, s1, s2, s3; } v4int3;
typedef struct { int4 s0, s1, s2, s3; } v4int4;
typedef struct { int8 s0, s1, s2, s3; } v4int8;
typedef struct { int16 s0, s1, s2, s3; } v4int16;



#define OCML_ATTR __attribute__((always_inline, const, overloadable))

#define ALIGNEDATTR(X) __attribute__((aligned(X)))
#define INLINEATTR __attribute__((always_inline))
#define PUREATTR __attribute__((pure))
#define CONSTATTR __attribute__((const))

#define FMA fma
#define RCP(X) ((vtype)(1.0) / X)
#define DIV(X,Y) (X / Y)

#define LDEXP ldexp
#define SQRT sqrt
#define ISINF isinf
#define COPYSIGN copysign
#define MATH_FAST_RCP RCP
#define MATH_RCP RCP
#define MATH_MAD pocl_fma

#define BUILTIN_ABS_F32 fabs
#define BUILTIN_TRUNC_F32 trunc
#define BUILTIN_FRACTION_F32 fract
#define BUILTIN_COPYSIGN_F32 copysign
#define BUILTIN_FMA_F32 fma

#define BUILTIN_FREXP_MANT_F32 _cl_frfrexp
#define BUILTIN_FLDEXP_F32 ldexp
#define BUILTIN_FREXP_EXP_F32 _cl_expfrexp
#define BUILTIN_RINT_F32 rint

#define BUILTIN_ABS_F64 fabs
#define BUILTIN_TRUNC_F64 trunc
#define BUILTIN_FRACTION_F64 fract
#define BUILTIN_COPYSIGN_F64 copysign
#define BUILTIN_FMA_F64 fma

#define BUILTIN_FREXP_MANT_F64 _cl_frfrexp
#define BUILTIN_FLDEXP_F64 ldexp
#define BUILTIN_FREXP_EXP_F64 _cl_expfrexp
#define BUILTIN_RINT_F64 rint

#define MATH_PRIVATE(NAME) __pocl_ ## NAME
#define MATH_MANGLE(NAME) _CL_OVERLOADABLE NAME

#ifndef _CL_DECLARE_FUNC_V_VVV
#define _CL_DECLARE_FUNC_V_VVV(NAME)                                    \
  __IF_FP16(                                                            \
  half     _CL_OVERLOADABLE NAME(half    , half    , half  );           \
  half2    _CL_OVERLOADABLE NAME(half2   , half2   , half2 );           \
  half3    _CL_OVERLOADABLE NAME(half3   , half3   , half3 );           \
  half4    _CL_OVERLOADABLE NAME(half4   , half4   , half4 );           \
  half8    _CL_OVERLOADABLE NAME(half8   , half8   , half8 );           \
  half16   _CL_OVERLOADABLE NAME(half16  , half16  , half16);)          \
  float    _CL_OVERLOADABLE NAME(float   , float   , float   );         \
  float2   _CL_OVERLOADABLE NAME(float2  , float2  , float2  );         \
  float3   _CL_OVERLOADABLE NAME(float3  , float3  , float3  );         \
  float4   _CL_OVERLOADABLE NAME(float4  , float4  , float4  );         \
  float8   _CL_OVERLOADABLE NAME(float8  , float8  , float8  );         \
  float16  _CL_OVERLOADABLE NAME(float16 , float16 , float16 );         \
  __IF_FP64(                                                            \
  double   _CL_OVERLOADABLE NAME(double  , double  , double  );         \
  double2  _CL_OVERLOADABLE NAME(double2 , double2 , double2 );         \
  double3  _CL_OVERLOADABLE NAME(double3 , double3 , double3 );         \
  double4  _CL_OVERLOADABLE NAME(double4 , double4 , double4 );         \
  double8  _CL_OVERLOADABLE NAME(double8 , double8 , double8 );         \
  double16 _CL_OVERLOADABLE NAME(double16, double16, double16);)
#endif

_CL_DECLARE_FUNC_V_VVV(pocl_fma)
