/* OpenCL built-in library: implementation templates

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

#define IMPLEMENT_BUILTIN_V_V(NAME, VTYPE, LO, HI)      \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a)                                         \
  {                                                     \
    return (VTYPE)(NAME(a.LO), NAME(a.HI));             \
  }
#define DEFINE_BUILTIN_V_V(NAME)                \
  float __attribute__ ((overloadable))          \
  NAME(float a)                                 \
  {                                             \
    return __builtin_##NAME##f(a);              \
  }                                             \
  IMPLEMENT_BUILTIN_V_V(NAME, float2  , lo, hi) \
  IMPLEMENT_BUILTIN_V_V(NAME, float4  , lo, hi) \
  IMPLEMENT_BUILTIN_V_V(NAME, float3  , lo, s2) \
  IMPLEMENT_BUILTIN_V_V(NAME, float8  , lo, hi) \
  IMPLEMENT_BUILTIN_V_V(NAME, float16 , lo, hi) \
  __IF_FP64(                                    \
  double __attribute__ ((overloadable))         \
  NAME(double a)                                \
  {                                             \
    return __builtin_##NAME(a);                 \
  }                                             \
  IMPLEMENT_BUILTIN_V_V(NAME, double2 , lo, hi) \
  IMPLEMENT_BUILTIN_V_V(NAME, double3 , lo, s2) \
  IMPLEMENT_BUILTIN_V_V(NAME, double4 , lo, hi) \
  IMPLEMENT_BUILTIN_V_V(NAME, double8 , lo, hi) \
  IMPLEMENT_BUILTIN_V_V(NAME, double16, lo, hi))

#define IMPLEMENT_BUILTIN_V_VV(NAME, VTYPE, LO, HI)     \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, VTYPE b)                                \
  {                                                     \
    return (VTYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI)); \
  }
#define DEFINE_BUILTIN_V_VV(NAME)                       \
  float __attribute__ ((overloadable))                  \
  NAME(float a, float b)                                \
  {                                                     \
    return __builtin_##NAME##f(a, b);                   \
  }                                                     \
  IMPLEMENT_BUILTIN_V_VV(NAME, float2  , lo, hi)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, float3  , lo, s2)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, float4  , lo, hi)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, float8  , lo, hi)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, float16 , lo, hi)        \
  __IF_FP64(                                            \
  double __attribute__ ((overloadable))                 \
  NAME(double a, double b)                              \
  {                                                     \
    return __builtin_##NAME(a, b);                      \
  }                                                     \
  IMPLEMENT_BUILTIN_V_VV(NAME, double2 , lo, hi)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, double3 , lo, s2)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, double4 , lo, hi)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, double8 , lo, hi)        \
  IMPLEMENT_BUILTIN_V_VV(NAME, double16, lo, hi))

#define IMPLEMENT_BUILTIN_V_VVV(NAME, VTYPE, LO, HI)                    \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, VTYPE b, VTYPE c)                                       \
  {                                                                     \
    return (VTYPE)(NAME(a.LO, b.LO, c.LO), NAME(a.HI, b.HI, c.HI));     \
  }
#define DEFINE_BUILTIN_V_VVV(NAME)                      \
  float __attribute__ ((overloadable))                  \
  NAME(float a, float b, float c)                       \
  {                                                     \
    return __builtin_##NAME##f(a, b, c);                \
  }                                                     \
  IMPLEMENT_BUILTIN_V_VVV(NAME, float2  , lo, hi)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, float3  , lo, s2)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, float4  , lo, hi)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, float8  , lo, hi)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, float16 , lo, hi)       \
  __IF_FP64(                                            \
  double __attribute__ ((overloadable))                 \
  NAME(double a, double b, double c)                    \
  {                                                     \
    return __builtin_##NAME(a, b, c);                   \
  }                                                     \
  IMPLEMENT_BUILTIN_V_VVV(NAME, double2 , lo, hi)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, double3 , lo, s2)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, double4 , lo, hi)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, double8 , lo, hi)       \
  IMPLEMENT_BUILTIN_V_VVV(NAME, double16, lo, hi))

#define IMPLEMENT_BUILTIN_V_U(NAME, VTYPE, UTYPE, LO, HI)       \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(UTYPE a)                                                 \
  {                                                             \
    return (VTYPE)(NAME(a.LO), NAME(a.HI));                     \
  }
#define DEFINE_BUILTIN_V_U(NAME)                        \
  float __attribute__ ((overloadable))                  \
  NAME(uint a)                                          \
  {                                                     \
    return __builtin_##NAME##f(a);                      \
  }                                                     \
  IMPLEMENT_BUILTIN_V_U(NAME, float2  , uint2  , lo, hi) \
  IMPLEMENT_BUILTIN_V_U(NAME, float3  , uint3  , lo, s2) \
  IMPLEMENT_BUILTIN_V_U(NAME, float4  , uint4  , lo, hi) \
  IMPLEMENT_BUILTIN_V_U(NAME, float8  , uint8  , lo, hi) \
  IMPLEMENT_BUILTIN_V_U(NAME, float16 , uint16 , lo, hi) \
  __IF_FP64(                                             \
  double __attribute__ ((overloadable))                 \
  NAME(ulong a)                                         \
  {                                                     \
    return __builtin_##NAME(a);                         \
  }                                                     \
  IMPLEMENT_BUILTIN_V_U(NAME, double2 , ulong2 , lo, hi) \
  IMPLEMENT_BUILTIN_V_U(NAME, double3 , ulong3 , lo, s2) \
  IMPLEMENT_BUILTIN_V_U(NAME, double4 , ulong4 , lo, hi) \
  IMPLEMENT_BUILTIN_V_U(NAME, double8 , ulong8 , lo, hi) \
  IMPLEMENT_BUILTIN_V_U(NAME, double16, ulong16, lo, hi))

#define IMPLEMENT_BUILTIN_J_VV(NAME, VTYPE, JTYPE, LO, HI)      \
  JTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, VTYPE b)                                        \
  {                                                             \
    return (JTYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI));         \
  }
#define DEFINE_BUILTIN_J_VV(NAME)                               \
  int __attribute__ ((overloadable))                            \
  NAME(float a, float b)                                        \
  {                                                             \
    return __builtin_##NAME##f(a, b);                           \
  }                                                             \
  IMPLEMENT_BUILTIN_J_VV(NAME, float2  , int2  , lo, hi)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, float3  , int3  , lo, s2)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, float4  , int4  , lo, hi)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, float8  , int8  , lo, hi)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, float16 , int16 , lo, hi)        \
  __IF_FP64(                                                    \
  int __attribute__ ((overloadable))                            \
  NAME(double a, double b)                                      \
  {                                                             \
    return __builtin_##NAME(a, b);                              \
  }                                                             \
  __IF_INT64(                                                   \
  IMPLEMENT_BUILTIN_J_VV(NAME, double2 , long2 , lo, hi)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, double3 , long3 , lo, s2)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, double4 , long4 , lo, hi)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, double8 , long8 , lo, hi)        \
  IMPLEMENT_BUILTIN_J_VV(NAME, double16, long16, lo, hi)))

#define IMPLEMENT_BUILTIN_L_VV(NAME, VTYPE, STYPE, LTYPE, LO, HI)       \
  LTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a, VTYPE b)                                                \
  {                                                                     \
    /* change sign? */                                                  \
    int cslo = sizeof(a.LO)==sizeof(STYPE);                             \
    int cshi = sizeof(a.HI)==sizeof(STYPE);                             \
    return (LTYPE)                                                      \
      (cslo ? -NAME(a.LO, b.LO) : NAME(a.LO, b.LO),                     \
       cshi ? -NAME(a.HI, b.HI) : NAME(a.HI, b.HI));                    \
  }
#define DEFINE_BUILTIN_L_VV(NAME)                                       \
  int __attribute__ ((overloadable))                                    \
  NAME(float a, float b)                                                \
  {                                                                     \
    return __builtin_##NAME##f(a, b);                                   \
  }                                                                     \
  IMPLEMENT_BUILTIN_L_VV(NAME, float2  , float , int2  , lo, hi)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, float3  , float , int3  , lo, s2)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, float4  , float , int4  , lo, hi)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, float8  , float , int8  , lo, hi)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, float16 , float , int16 , lo, hi)        \
  __IF_FP64(                                                            \
  int __attribute__ ((overloadable))                                    \
  NAME(double a, double b)                                              \
  {                                                                     \
    return __builtin_##NAME(a, b);                                      \
  }                                                                     \
  __IF_INT64(                                                           \
  IMPLEMENT_BUILTIN_L_VV(NAME, double2 , double, long2 , lo, hi)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, double3 , double, long3 , lo, s2)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, double4 , double, long4 , lo, hi)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, double8 , double, long8 , lo, hi)        \
  IMPLEMENT_BUILTIN_L_VV(NAME, double16, double, long16, lo, hi)))

#define IMPLEMENT_BUILTIN_V_VJ(NAME, VTYPE, JTYPE, LO, HI)      \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, JTYPE b)                                        \
  {                                                             \
    return (VTYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI));         \
  }
#define DEFINE_BUILTIN_V_VJ(NAME)                       \
  float __attribute__ ((overloadable))                  \
  NAME(float a, int b)                                  \
  {                                                     \
    return __builtin_##NAME##f(a, b);                   \
  }                                                     \
  IMPLEMENT_BUILTIN_V_VJ(NAME, float2  , int2 , lo, hi) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, float3  , int3 , lo, s2) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, float4  , int4 , lo, hi) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, float8  , int8 , lo, hi) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, float16 , int16, lo, hi) \
  __IF_FP64(                                            \
  double __attribute__ ((overloadable))                 \
  NAME(double a, int b)                                 \
  {                                                     \
    return __builtin_##NAME(a, b);                      \
  }                                                     \
  IMPLEMENT_BUILTIN_V_VJ(NAME, double2 , int2 , lo, hi) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, double3 , int3 , lo, s2) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, double4 , int4 , lo, hi) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, double8 , int8 , lo, hi) \
  IMPLEMENT_BUILTIN_V_VJ(NAME, double16, int16, lo, hi))

#define IMPLEMENT_BUILTIN_V_VI(NAME, VTYPE, ITYPE, LO, HI)      \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, ITYPE b)                                        \
  {                                                             \
    return (VTYPE)(NAME(a.LO, b), NAME(a.HI, b));               \
  }
#define DEFINE_BUILTIN_V_VI(NAME)                       \
  IMPLEMENT_BUILTIN_V_VI(NAME, float2  , int, lo, hi)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, float3  , int, lo, s2)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, float4  , int, lo, hi)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, float8  , int, lo, hi)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, float16 , int, lo, hi)   \
  __IF_FP64(                                            \
  IMPLEMENT_BUILTIN_V_VI(NAME, double2 , int, lo, hi)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, double3 , int, lo, s2)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, double4 , int, lo, hi)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, double8 , int, lo, hi)   \
  IMPLEMENT_BUILTIN_V_VI(NAME, double16, int, lo, hi))

#define IMPLEMENT_BUILTIN_J_V(NAME, JTYPE, VTYPE, LO, HI)       \
  JTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a)                                                 \
  {                                                             \
    return (JTYPE)(NAME(a.LO), NAME(a.HI));                     \
  }
#define DEFINE_BUILTIN_J_V(NAME)                        \
  int __attribute__ ((overloadable))                    \
  NAME(float a)                                         \
  {                                                     \
    return __builtin_##NAME##f(a);                      \
  }                                                     \
  IMPLEMENT_BUILTIN_J_V(NAME, int2 , float2  , lo, hi)  \
  IMPLEMENT_BUILTIN_J_V(NAME, int3 , float3  , lo, s2)  \
  IMPLEMENT_BUILTIN_J_V(NAME, int4 , float4  , lo, hi)  \
  IMPLEMENT_BUILTIN_J_V(NAME, int8 , float8  , lo, hi)  \
  IMPLEMENT_BUILTIN_J_V(NAME, int16, float16 , lo, hi)  \
  __IF_FP64(                                            \
  int __attribute__ ((overloadable))                    \
  NAME(double a)                                        \
  {                                                     \
    return __builtin_##NAME(a);                         \
  }                                                     \
  IMPLEMENT_BUILTIN_J_V(NAME, long2 , double2 , lo, hi) \
  IMPLEMENT_BUILTIN_J_V(NAME, long3 , double3 , lo, s2) \
  IMPLEMENT_BUILTIN_J_V(NAME, long4 , double4 , lo, hi) \
  IMPLEMENT_BUILTIN_J_V(NAME, long8 , double8 , lo, hi) \
  IMPLEMENT_BUILTIN_J_V(NAME, long16, double16, lo, hi))

#define IMPLEMENT_BUILTIN_K_V(NAME, JTYPE, VTYPE, LO, HI)       \
  JTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a)                                                 \
  {                                                             \
    return (JTYPE)(NAME(a.LO), NAME(a.HI));                     \
  }
#define DEFINE_BUILTIN_K_V(NAME)                        \
  int __attribute__ ((overloadable))                    \
  NAME(float a)                                         \
  {                                                     \
    return __builtin_##NAME##f(a);                      \
  }                                                     \
  IMPLEMENT_BUILTIN_K_V(NAME, int2  , float2  , lo, hi) \
  IMPLEMENT_BUILTIN_K_V(NAME, int3  , float3  , lo, s2) \
  IMPLEMENT_BUILTIN_K_V(NAME, int4  , float4  , lo, hi) \
  IMPLEMENT_BUILTIN_K_V(NAME, int8  , float8  , lo, hi) \
  IMPLEMENT_BUILTIN_K_V(NAME, int16 , float16 , lo, hi) \
  __IF_FP64(                                            \
  int __attribute__ ((overloadable))                    \
  NAME(double a)                                        \
  {                                                     \
    return __builtin_##NAME(a);                         \
  }                                                     \
  IMPLEMENT_BUILTIN_K_V(NAME, int2 , double2 , lo, hi)  \
  IMPLEMENT_BUILTIN_K_V(NAME, int3 , double3 , lo, s2)  \
  IMPLEMENT_BUILTIN_K_V(NAME, int4 , double4 , lo, hi)  \
  IMPLEMENT_BUILTIN_K_V(NAME, int8 , double8 , lo, hi)  \
  IMPLEMENT_BUILTIN_K_V(NAME, int16, double16, lo, hi))

#define IMPLEMENT_BUILTIN_L_V(NAME, LTYPE, VTYPE, STYPE, LO, HI)        \
  LTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a)                                                         \
  {                                                                     \
    /* change sign? */                                                  \
    int cslo = sizeof(a.LO)==sizeof(STYPE);                             \
    int cshi = sizeof(a.HI)==sizeof(STYPE);                             \
    return (LTYPE)                                                      \
      (cslo ? -NAME(a.LO) : NAME(a.LO),                                 \
       cshi ? -NAME(a.HI) : NAME(a.HI));                                \
  }
#define DEFINE_BUILTIN_L_V(NAME)                                        \
  int __attribute__ ((overloadable))                                    \
  NAME(float a)                                                         \
  {                                                                     \
    return __builtin_##NAME##f(a);                                      \
  }                                                                     \
  IMPLEMENT_BUILTIN_L_V(NAME, int2  , float2  , float , lo, hi)         \
  IMPLEMENT_BUILTIN_L_V(NAME, int3  , float3  , float , lo, s2)         \
  IMPLEMENT_BUILTIN_L_V(NAME, int4  , float4  , float , lo, hi)         \
  IMPLEMENT_BUILTIN_L_V(NAME, int8  , float8  , float , lo, hi)         \
  IMPLEMENT_BUILTIN_L_V(NAME, int16 , float16 , float , lo, hi)         \
  __IF_FP64(                                                            \
  int __attribute__ ((overloadable))                                    \
  NAME(double a)                                                        \
  {                                                                     \
    return __builtin_##NAME(a);                                         \
  }                                                                     \
  IMPLEMENT_BUILTIN_L_V(NAME, long2 , double2 , double, lo, hi)         \
  IMPLEMENT_BUILTIN_L_V(NAME, long3 , double3 , double, lo, s2)         \
  IMPLEMENT_BUILTIN_L_V(NAME, long4 , double4 , double, lo, hi)         \
  IMPLEMENT_BUILTIN_L_V(NAME, long8 , double8 , double, lo, hi)         \
  IMPLEMENT_BUILTIN_L_V(NAME, long16, double16, double, lo, hi))

/******************************************************************************/

#define IMPLEMENT_EXPR_V_V(NAME, EXPR, VTYPE, STYPE, JTYPE, SJTYPE)     \
  VTYPE __attribute__ ((overloadable))                                  \
  NAME(VTYPE a)                                                         \
  {                                                                     \
    typedef VTYPE vtype;                                                \
    typedef STYPE stype;                                                \
    typedef JTYPE jtype;                                                \
    typedef SJTYPE sjtype;                                              \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_V_V(NAME, EXPR)                                     \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, float   , float , int   , int )        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, float2  , float , int2  , int )        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, float3  , float , int3  , int )        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, float4  , float , int4  , int )        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, float8  , float , int8  , int )        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, float16 , float , int16 , int )        \
  __IF_FP64(                                                            \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, double  , double, long  , long)        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, double2 , double, long2 , long)        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, double3 , double, long3 , long)        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, double4 , double, long4 , long)        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, double8 , double, long8 , long)        \
  IMPLEMENT_EXPR_V_V(NAME, EXPR, double16, double, long16, long))

#define IMPLEMENT_EXPR_V_VV(NAME, EXPR, VTYPE, STYPE, JTYPE)    \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, VTYPE b)                                        \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef JTYPE jtype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_V_VV(NAME, EXPR)                            \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, float   , float , int   )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, float2  , float , int2  )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, float3  , float , int3  )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, float4  , float , int4  )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, float8  , float , int8  )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, float16 , float , int16 )     \
  __IF_FP64(                                                    \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, double  , double, long  )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, double2 , double, long2 )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, double3 , double, long3 )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, double4 , double, long4 )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, double8 , double, long8 )     \
  IMPLEMENT_EXPR_V_VV(NAME, EXPR, double16, double, long16))

#define IMPLEMENT_EXPR_V_VVV(NAME, EXPR, VTYPE, STYPE, JTYPE)   \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, VTYPE b, VTYPE c)                               \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef JTYPE jtype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_V_VVV(NAME, EXPR)                           \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, float   , float , int   )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, float2  , float , int2  )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, float3  , float , int3  )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, float4  , float , int4  )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, float8  , float , int8  )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, float16 , float , int16 )    \
  __IF_FP64(                                                    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, double  , double, long  )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, double2 , double, long2 )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, double3 , double, long3 )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, double4 , double, long4 )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, double8 , double, long8 )    \
  IMPLEMENT_EXPR_V_VVV(NAME, EXPR, double16, double, long16))

#define IMPLEMENT_EXPR_S_V(NAME, EXPR, VTYPE, STYPE)		\
  STYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a)							\
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_S_V(NAME, EXPR)                     \
  IMPLEMENT_EXPR_S_V(NAME, EXPR, float   , float )      \
  IMPLEMENT_EXPR_S_V(NAME, EXPR, float2  , float )      \
  IMPLEMENT_EXPR_S_V(NAME, EXPR, float3  , float )      \
  IMPLEMENT_EXPR_S_V(NAME, EXPR, float4  , float )      \
  IMPLEMENT_EXPR_S_V(NAME, EXPR, float8  , float )      \
  IMPLEMENT_EXPR_S_V(NAME, EXPR, float16 , float )      \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_S_V(NAME, EXPR, double  , double )	\
  IMPLEMENT_EXPR_S_V(NAME, EXPR, double2 , double )	\
  IMPLEMENT_EXPR_S_V(NAME, EXPR, double3 , double )	\
  IMPLEMENT_EXPR_S_V(NAME, EXPR, double4 , double )	\
  IMPLEMENT_EXPR_S_V(NAME, EXPR, double8 , double )	\
  IMPLEMENT_EXPR_S_V(NAME, EXPR, double16, double ))

#define IMPLEMENT_EXPR_S_VV(NAME, EXPR, VTYPE, STYPE, JTYPE)    \
  STYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, VTYPE b)                                        \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef JTYPE jtype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_S_VV(NAME, EXPR)                            \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, float   , float , int   )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, float2  , float , int2  )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, float3  , float , int3  )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, float4  , float , int4  )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, float8  , float , int8  )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, float16 , float , int16 )     \
  __IF_FP64(                                                    \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, double  , double, long  )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, double2 , double, long2 )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, double3 , double, long3 )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, double4 , double, long4 )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, double8 , double, long8 )     \
  IMPLEMENT_EXPR_S_VV(NAME, EXPR, double16, double, long16))

#define IMPLEMENT_EXPR_J_V(NAME, EXPR, VTYPE, JTYPE)    \
  JTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a)                                         \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef JTYPE jtype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_J_V(NAME, EXPR)                     \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, float   , int   )      \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, float2  , int2  )      \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, float3  , int3  )      \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, float4  , int4  )      \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, float8  , int8  )      \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, float16 , int16 )      \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, double  , int   ))	\
  __IF_INT64(                                           \
  IMPLEMENT_EXPR_J_V(NAME, EXPR, double2 , long2 )	\
  IMPLEMENT_EXPR_J_V(NAME, EXPR, double3 , long3 )	\
  IMPLEMENT_EXPR_J_V(NAME, EXPR, double4 , long4 )	\
  IMPLEMENT_EXPR_J_V(NAME, EXPR, double8 , long8 )	\
  IMPLEMENT_EXPR_J_V(NAME, EXPR, double16, long16))

#define IMPLEMENT_EXPR_J_VV(NAME, EXPR, VTYPE, JTYPE)   \
  JTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, VTYPE b)                                \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef JTYPE jtype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_J_VV(NAME, EXPR)                    \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, float   , int   )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, float2  , int2  )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, float3  , int3  )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, float4  , int4  )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, float8  , int8  )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, float16 , int16 )     \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, double  , int   ))    \
  __IF_INT64(                                           \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, double2 , long2 )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, double3 , long3 )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, double4 , long4 )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, double8 , long8 )     \
  IMPLEMENT_EXPR_J_VV(NAME, EXPR, double16, long16))

#define IMPLEMENT_EXPR_V_VVS(NAME, EXPR, VTYPE, STYPE)  \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, VTYPE b, STYPE c)                       \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_V_VVS(NAME, EXPR)                   \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, float2  , float )    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, float3  , float )    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, float4  , float )    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, float8  , float )    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, float16 , float )    \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, double2 , double)    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, double3 , double)    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, double4 , double)    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, double8 , double)    \
  IMPLEMENT_EXPR_V_VVS(NAME, EXPR, double16, double))

#define IMPLEMENT_EXPR_V_VSS(NAME, EXPR, VTYPE, STYPE)  \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, STYPE b, STYPE c)                       \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_V_VSS(NAME, EXPR)                   \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, float2  , float )    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, float3  , float )    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, float4  , float )    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, float8  , float )    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, float16 , float )    \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, double2 , double)    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, double3 , double)    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, double4 , double)    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, double8 , double)    \
  IMPLEMENT_EXPR_V_VSS(NAME, EXPR, double16, double))

#define IMPLEMENT_EXPR_V_SSV(NAME, EXPR, VTYPE, STYPE)  \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(STYPE a, STYPE b, VTYPE c)                       \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_V_SSV(NAME, EXPR)                   \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, float2  , float )    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, float3  , float )    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, float4  , float )    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, float8  , float )    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, float16 , float )    \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, double2 , double)    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, double3 , double)    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, double4 , double)    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, double8 , double)    \
  IMPLEMENT_EXPR_V_SSV(NAME, EXPR, double16, double))

#define IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, VTYPE, STYPE, JTYPE)   \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, VTYPE b, JTYPE c)                               \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef JTYPE jtype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_V_VVJ(NAME, EXPR)                           \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, float   , float , int   )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, float2  , float , int2  )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, float3  , float , int3  )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, float4  , float , int4  )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, float8  , float , int8  )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, float16 , float , int16 )    \
  __IF_INT64(__IF_FP64(                                         \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, double  , double, long  )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, double2 , double, long2 )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, double3 , double, long3 )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, double4 , double, long4 )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, double8 , double, long8 )    \
  IMPLEMENT_EXPR_V_VVJ(NAME, EXPR, double16, double, long16)))

#define IMPLEMENT_EXPR_V_U(NAME, EXPR, VTYPE, STYPE, UTYPE)     \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(UTYPE a)                                                 \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef UTYPE utype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_V_U(NAME, EXPR)                             \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, float   , float , uint   )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, float2  , float , uint2  )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, float3  , float , uint3  )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, float4  , float , uint4  )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, float8  , float , uint8  )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, float16 , float , uint16 )     \
  __IF_INT64(__IF_FP64(                                         \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, double  , double, ulong  )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, double2 , double, ulong2 )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, double3 , double, ulong3 )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, double4 , double, ulong4 )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, double8 , double, ulong8 )     \
  IMPLEMENT_EXPR_V_U(NAME, EXPR, double16, double, ulong16)))

#define IMPLEMENT_EXPR_V_VS(NAME, EXPR, VTYPE, STYPE)   \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, STYPE b)                                \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_V_VS(NAME, EXPR)                    \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, float2  , float )     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, float3  , float )     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, float4  , float )     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, float8  , float )     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, float16 , float )     \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, double2 , double)     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, double3 , double)     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, double4 , double)     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, double8 , double)     \
  IMPLEMENT_EXPR_V_VS(NAME, EXPR, double16, double))

#define IMPLEMENT_EXPR_V_VJ(NAME, EXPR, VTYPE, STYPE, JTYPE)    \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, JTYPE b)                                        \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef JTYPE jtype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_V_VJ(NAME, EXPR)                            \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, float   , float , int  )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, float2  , float , int2 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, float3  , float , int3 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, float4  , float , int4 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, float8  , float , int8 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, float16 , float , int16)      \
  __IF_FP64(                                                    \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, double  , double, int  )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, double2 , double, int2 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, double3 , double, int3 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, double4 , double, int4 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, double8 , double, int8 )      \
  IMPLEMENT_EXPR_V_VJ(NAME, EXPR, double16, double, int16))

#define IMPLEMENT_EXPR_V_VI(NAME, EXPR, VTYPE, STYPE, ITYPE)    \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, ITYPE b)                                        \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef ITYPE itype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_V_VI(NAME, EXPR)                            \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, float2  , float , int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, float3  , float , int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, float4  , float , int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, float8  , float , int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, float16 , float , int)        \
  __IF_FP64(                                                    \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, double2 , double, int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, double3 , double, int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, double4 , double, int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, double8 , double, int)        \
  IMPLEMENT_EXPR_V_VI(NAME, EXPR, double16, double, int))

#define IMPLEMENT_EXPR_V_VPV(NAME, EXPR, VTYPE, STYPE)  \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, __global VTYPE *b)                      \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }                                                     \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, __local VTYPE *b)                       \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }                                                     \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a, __private VTYPE *b)                     \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_V_VPV(NAME, EXPR)                   \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, float   , float )    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, float2  , float )    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, float3  , float )    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, float4  , float )    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, float8  , float )    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, float16 , float )    \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, double  , double)    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, double2 , double)    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, double3 , double)    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, double4 , double)    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, double8 , double)    \
  IMPLEMENT_EXPR_V_VPV(NAME, EXPR, double16, double))

#define IMPLEMENT_EXPR_V_SV(NAME, EXPR, VTYPE, STYPE)   \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(STYPE a, VTYPE b)                                \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_V_SV(NAME, EXPR)                    \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, float2  , float )     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, float3  , float )     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, float4  , float )     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, float8  , float )     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, float16 , float )     \
  __IF_FP64(                                            \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, double2 , double)     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, double3 , double)     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, double4 , double)     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, double8 , double)     \
  IMPLEMENT_EXPR_V_SV(NAME, EXPR, double16, double))

#define IMPLEMENT_EXPR_F_F(NAME, EXPR, VTYPE, STYPE)    \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a)                                         \
  {                                                     \
    typedef VTYPE vtype;                                \
    typedef STYPE stype;                                \
    return EXPR;                                        \
  }
#define DEFINE_EXPR_F_F(NAME, EXPR)                     \
  IMPLEMENT_EXPR_F_F(NAME, EXPR, float   , float )      \
  IMPLEMENT_EXPR_F_F(NAME, EXPR, float2  , float )      \
  IMPLEMENT_EXPR_F_F(NAME, EXPR, float3  , float )      \
  IMPLEMENT_EXPR_F_F(NAME, EXPR, float4  , float )      \
  IMPLEMENT_EXPR_F_F(NAME, EXPR, float8  , float )      \
  IMPLEMENT_EXPR_F_F(NAME, EXPR, float16 , float )

#define IMPLEMENT_EXPR_F_FF(NAME, EXPR, VTYPE, STYPE, JTYPE)    \
  VTYPE __attribute__ ((overloadable))                          \
  NAME(VTYPE a, VTYPE b)                                        \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef JTYPE jtype;                                        \
    return EXPR;                                                \
  }
#define DEFINE_EXPR_F_FF(NAME, EXPR)                            \
  IMPLEMENT_EXPR_F_FF(NAME, EXPR, float   , float , int   )     \
  IMPLEMENT_EXPR_F_FF(NAME, EXPR, float2  , float , int2  )     \
  IMPLEMENT_EXPR_F_FF(NAME, EXPR, float3  , float , int3  )     \
  IMPLEMENT_EXPR_F_FF(NAME, EXPR, float4  , float , int4  )     \
  IMPLEMENT_EXPR_F_FF(NAME, EXPR, float8  , float , int8  )     \
  IMPLEMENT_EXPR_F_FF(NAME, EXPR, float16 , float , int16 )



#define IMPLEMENT_BUILTIN_G_G(NAME, GTYPE, UGTYPE, LO, HI)      \
  GTYPE __attribute__ ((overloadable))                          \
  NAME(GTYPE a)                                                 \
  {                                                             \
    return (GTYPE)(NAME(a.LO), NAME(a.HI));                     \
  }
#define DEFINE_BUILTIN_G_G(NAME)                                \
  char __attribute__ ((overloadable))                           \
  NAME(char a)                                                  \
  {                                                             \
    return __builtin_##NAME##hh(a);                             \
  }                                                             \
  uchar __attribute__ ((overloadable))                          \
  NAME(uchar a)                                                 \
  {                                                             \
    return __builtin_##NAME##uhh(a);                            \
  }                                                             \
  short __attribute__ ((overloadable))                          \
  NAME(short a)                                                 \
  {                                                             \
    return __builtin_##NAME##h(a);                              \
  }                                                             \
  ushort __attribute__ ((overloadable))                         \
  NAME(ushort a)                                                \
  {                                                             \
    return __builtin_##NAME##uh(a);                             \
  }                                                             \
  int __attribute__ ((overloadable))                            \
  NAME(int a)                                                   \
  {                                                             \
    return __builtin_##NAME(a);                                 \
  }                                                             \
  uint __attribute__ ((overloadable))                           \
  NAME(uint a)                                                  \
  {                                                             \
    return __builtin_##NAME##u(a);                              \
  }                                                             \
  __IF_INT64(                                                   \
  long __attribute__ ((overloadable))                           \
  NAME(long a)                                                  \
  {                                                             \
    return __builtin_##NAME##l(a);                              \
  }                                                             \
  ulong __attribute__ ((overloadable))                          \
  NAME(ulong a)                                                 \
  {                                                             \
    return __builtin_##NAME##ul(a);                             \
  })                                                            \
  IMPLEMENT_BUILTIN_G_G(NAME, char2   , uchar2  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, char3   , uchar3  , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, char4   , uchar4  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, char8   , uchar8  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, char16  , uchar16 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uchar2  , uchar2  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uchar3  , uchar3  , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uchar4  , uchar4  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uchar8  , uchar8  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uchar16 , uchar16 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, short2  , ushort2 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, short3  , ushort3 , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, short4  , ushort4 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, short8  , ushort8 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, short16 , ushort16, lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ushort2 , ushort2 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ushort3 , ushort3 , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ushort4 , ushort4 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ushort8 , ushort8 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ushort16, ushort16, lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, int2    , uint2   , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, int3    , uint3   , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, int4    , uint4   , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, int8    , uint8   , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, int16   , uint16  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uint2   , uint2   , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uint3   , uint3   , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uint4   , uint4   , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uint8   , uint8   , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, uint16  , uint16  , lo, hi)       \
  __IF_INT64(                                                   \
  IMPLEMENT_BUILTIN_G_G(NAME, long2   , ulong2  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, long3   , ulong3  , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, long4   , ulong4  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, long8   , ulong8  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, long16  , ulong16 , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ulong2  , ulong2  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ulong3  , ulong3  , lo, s2)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ulong4  , ulong4  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ulong8  , ulong8  , lo, hi)       \
  IMPLEMENT_BUILTIN_G_G(NAME, ulong16 , ulong16 , lo, hi))

#define IMPLEMENT_BUILTIN_UG_G(NAME, GTYPE, UGTYPE, LO, HI)     \
  UGTYPE __attribute__ ((overloadable))                         \
  NAME(GTYPE a)                                                 \
  {                                                             \
    return (UGTYPE)(NAME(a.LO), NAME(a.HI));                    \
  }
#define DEFINE_BUILTIN_UG_G(NAME)                               \
  uchar __attribute__ ((overloadable))                          \
  NAME(char a)                                                  \
  {                                                             \
    return __builtin_##NAME##hh(a);                             \
  }                                                             \
  uchar __attribute__ ((overloadable))                          \
  NAME(uchar a)                                                 \
  {                                                             \
    return __builtin_##NAME##uhh(a);                            \
  }                                                             \
  ushort __attribute__ ((overloadable))                         \
  NAME(short a)                                                 \
  {                                                             \
    return __builtin_##NAME##h(a);                              \
  }                                                             \
  ushort __attribute__ ((overloadable))                         \
  NAME(ushort a)                                                \
  {                                                             \
    return __builtin_##NAME##uh(a);                             \
  }                                                             \
  uint __attribute__ ((overloadable))                           \
  NAME(int a)                                                   \
  {                                                             \
    return __builtin_##NAME(a);                                 \
  }                                                             \
  uint __attribute__ ((overloadable))                           \
  NAME(uint a)                                                  \
  {                                                             \
    return __builtin_##NAME##u(a);                              \
  }                                                             \
  __IF_INT64(                                                   \
  ulong __attribute__ ((overloadable))                          \
  NAME(long a)                                                  \
  {                                                             \
    return __builtin_##NAME##l(a);                              \
  }                                                             \
  ulong __attribute__ ((overloadable))                          \
  NAME(ulong a)                                                 \
  {                                                             \
    return __builtin_##NAME##ul(a);                             \
  })                                                            \
  IMPLEMENT_BUILTIN_UG_G(NAME, char2   , uchar2  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, char3   , uchar3  , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, char4   , uchar4  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, char8   , uchar8  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, char16  , uchar16 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uchar2  , uchar2  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uchar3  , uchar3  , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uchar4  , uchar4  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uchar8  , uchar8  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uchar16 , uchar16 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, short2  , ushort2 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, short3  , ushort3 , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, short4  , ushort4 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, short8  , ushort8 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, short16 , ushort16, lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ushort2 , ushort2 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ushort3 , ushort3 , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ushort4 , ushort4 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ushort8 , ushort8 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ushort16, ushort16, lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, int2    , uint2   , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, int3    , uint3   , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, int4    , uint4   , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, int8    , uint8   , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, int16   , uint16  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uint2   , uint2   , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uint3   , uint3   , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uint4   , uint4   , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uint8   , uint8   , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, uint16  , uint16  , lo, hi)      \
  __IF_INT64(                                                   \
  IMPLEMENT_BUILTIN_UG_G(NAME, long2   , ulong2  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, long3   , ulong3  , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, long4   , ulong4  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, long8   , ulong8  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, long16  , ulong16 , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ulong2  , ulong2  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ulong3  , ulong3  , lo, s2)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ulong4  , ulong4  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ulong8  , ulong8  , lo, hi)      \
  IMPLEMENT_BUILTIN_UG_G(NAME, ulong16 , ulong16 , lo, hi))

#define IMPLEMENT_EXPR_G_G(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, SUGTYPE)  \
  GTYPE __attribute__ ((overloadable))                                  \
  NAME(GTYPE a)                                                         \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_G_G(NAME, EXPR)                                     \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, char    , char  , uchar   , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, char2   , char  , uchar2  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, char3   , char  , uchar3  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, char4   , char  , uchar4  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, char8   , char  , uchar8  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, char16  , char  , uchar16 , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uchar   , uchar , uchar   , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, short   , short , ushort  , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, short2  , short , ushort2 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, short3  , short , ushort3 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, short4  , short , ushort4 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, short8  , short , ushort8 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, short16 , short , ushort16, ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ushort  , ushort, ushort  , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ushort16, ushort, ushort16, ushort)    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, int     , int   , uint    , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, int2    , int   , uint2   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, int3    , int   , uint3   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, int4    , int   , uint4   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, int8    , int   , uint8   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, int16   , int   , uint16  , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uint    , uint  , uint    , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uint2   , uint  , uint2   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uint3   , uint  , uint3   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uint4   , uint  , uint4   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uint8   , uint  , uint8   , uint  )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, uint16  , uint  , uint16  , uint  )    \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, long    , long  , ulong   , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, long2   , long  , ulong2  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, long3   , long  , ulong3  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, long4   , long  , ulong4  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, long8   , long  , ulong8  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, long16  , long  , ulong16 , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ulong   , ulong , ulong   , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )    \
  IMPLEMENT_EXPR_G_G(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define IMPLEMENT_EXPR_UG_G(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, SUGTYPE) \
  UGTYPE __attribute__ ((overloadable))                                 \
  NAME(GTYPE a)                                                         \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_UG_G(NAME, EXPR)                                    \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, char    , char  , uchar   , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, char2   , char  , uchar2  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, char3   , char  , uchar3  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, char4   , char  , uchar4  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, char8   , char  , uchar8  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, char16  , char  , uchar16 , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uchar   , uchar , uchar   , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, short   , short , ushort  , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, short2  , short , ushort2 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, short3  , short , ushort3 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, short4  , short , ushort4 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, short8  , short , ushort8 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, short16 , short , ushort16, ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ushort  , ushort, ushort  , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ushort16, ushort, ushort16, ushort)   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, int     , int   , uint    , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, int2    , int   , uint2   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, int3    , int   , uint3   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, int4    , int   , uint4   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, int8    , int   , uint8   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, int16   , int   , uint16  , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uint    , uint  , uint    , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uint2   , uint  , uint2   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uint3   , uint  , uint3   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uint4   , uint  , uint4   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uint8   , uint  , uint8   , uint  )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, uint16  , uint  , uint16  , uint  )   \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, long    , long  , ulong   , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, long2   , long  , ulong2  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, long3   , long  , ulong3  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, long4   , long  , ulong4  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, long8   , long  , ulong8  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, long16  , long  , ulong16 , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ulong   , ulong , ulong   , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )   \
  IMPLEMENT_EXPR_UG_G(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define IMPLEMENT_EXPR_G_GG(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, SUGTYPE) \
  GTYPE __attribute__ ((overloadable))                                  \
  NAME(GTYPE a, GTYPE b)                                                \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_G_GG(NAME, EXPR)                                    \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, char    , char  , uchar   , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, char2   , char  , uchar2  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, char3   , char  , uchar3  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, char4   , char  , uchar4  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, char8   , char  , uchar8  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, char16  , char  , uchar16 , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uchar   , uchar , uchar   , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, short   , short , ushort  , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, short2  , short , ushort2 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, short3  , short , ushort3 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, short4  , short , ushort4 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, short8  , short , ushort8 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, short16 , short , ushort16, ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ushort  , ushort, ushort  , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ushort16, ushort, ushort16, ushort)   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, int     , int   , uint    , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, int2    , int   , uint2   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, int3    , int   , uint3   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, int4    , int   , uint4   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, int8    , int   , uint8   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, int16   , int   , uint16  , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uint    , uint  , uint    , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uint2   , uint  , uint2   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uint3   , uint  , uint3   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uint4   , uint  , uint4   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uint8   , uint  , uint8   , uint  )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, uint16  , uint  , uint16  , uint  )   \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, long    , long  , ulong   , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, long2   , long  , ulong2  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, long3   , long  , ulong3  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, long4   , long  , ulong4  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, long8   , long  , ulong8  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, long16  , long  , ulong16 , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ulong   , ulong , ulong   , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )   \
  IMPLEMENT_EXPR_G_GG(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define IMPLEMENT_EXPR_G_GGG(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, SUGTYPE) \
  GTYPE __attribute__ ((overloadable))                                  \
  NAME(GTYPE a, GTYPE b, GTYPE c)                                       \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_G_GGG(NAME, EXPR)                                   \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, char    , char  , uchar   , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, char2   , char  , uchar2  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, char3   , char  , uchar3  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, char4   , char  , uchar4  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, char8   , char  , uchar8  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, char16  , char  , uchar16 , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uchar   , uchar , uchar   , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, short   , short , ushort  , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, short2  , short , ushort2 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, short3  , short , ushort3 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, short4  , short , ushort4 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, short8  , short , ushort8 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, short16 , short , ushort16, ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ushort  , ushort, ushort  , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ushort16, ushort, ushort16, ushort)  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, int     , int   , uint    , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, int2    , int   , uint2   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, int3    , int   , uint3   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, int4    , int   , uint4   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, int8    , int   , uint8   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, int16   , int   , uint16  , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uint    , uint  , uint    , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uint2   , uint  , uint2   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uint3   , uint  , uint3   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uint4   , uint  , uint4   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uint8   , uint  , uint8   , uint  )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, uint16  , uint  , uint16  , uint  )  \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, long    , long  , ulong   , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, long2   , long  , ulong2  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, long3   , long  , ulong3  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, long4   , long  , ulong4  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, long8   , long  , ulong8  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, long16  , long  , ulong16 , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ulong   , ulong , ulong   , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )  \
  IMPLEMENT_EXPR_G_GGG(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define IMPLEMENT_EXPR_G_GS(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, SUGTYPE) \
  GTYPE __attribute__ ((overloadable))                                  \
  NAME(GTYPE a, SGTYPE b)                                               \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_G_GS(NAME, EXPR)                                    \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, char2   , char  , uchar2  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, char3   , char  , uchar3  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, char4   , char  , uchar4  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, char8   , char  , uchar8  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, char16  , char  , uchar16 , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, short2  , short , ushort2 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, short3  , short , ushort3 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, short4  , short , ushort4 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, short8  , short , ushort8 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, short16 , short , ushort16, ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ushort16, ushort, ushort16, ushort)   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, int2    , int   , uint2   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, int3    , int   , uint3   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, int4    , int   , uint4   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, int8    , int   , uint8   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, int16   , int   , uint16  , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uint2   , uint  , uint2   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uint3   , uint  , uint3   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uint4   , uint  , uint4   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uint8   , uint  , uint8   , uint  )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, uint16  , uint  , uint16  , uint  )   \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, long2   , long  , ulong2  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, long3   , long  , ulong3  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, long4   , long  , ulong4  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, long8   , long  , ulong8  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, long16  , long  , ulong16 , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )   \
  IMPLEMENT_EXPR_G_GS(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define IMPLEMENT_EXPR_G_GSS(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, SUGTYPE) \
  GTYPE __attribute__ ((overloadable))                                  \
  NAME(GTYPE a, SGTYPE b, SGTYPE c)                                     \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_G_GSS(NAME, EXPR)                                   \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, char2   , char  , uchar2  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, char3   , char  , uchar3  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, char4   , char  , uchar4  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, char8   , char  , uchar8  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, char16  , char  , uchar16 , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, short2  , short , ushort2 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, short3  , short , ushort3 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, short4  , short , ushort4 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, short8  , short , ushort8 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, short16 , short , ushort16, ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ushort16, ushort, ushort16, ushort)  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, int2    , int   , uint2   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, int3    , int   , uint3   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, int4    , int   , uint4   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, int8    , int   , uint8   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, int16   , int   , uint16  , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uint2   , uint  , uint2   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uint3   , uint  , uint3   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uint4   , uint  , uint4   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uint8   , uint  , uint8   , uint  )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, uint16  , uint  , uint16  , uint  )  \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, long2   , long  , ulong2  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, long3   , long  , ulong3  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, long4   , long  , ulong4  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, long8   , long  , ulong8  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, long16  , long  , ulong16 , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )  \
  IMPLEMENT_EXPR_G_GSS(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define IMPLEMENT_EXPR_UG_GG(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, SUGTYPE) \
  UGTYPE __attribute__ ((overloadable))                                 \
  NAME(GTYPE a, GTYPE b)                                                \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef SUGTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_UG_GG(NAME, EXPR)                                   \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, char    , char  , uchar   , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, char2   , char  , uchar2  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, char3   , char  , uchar3  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, char4   , char  , uchar4  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, char8   , char  , uchar8  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, char16  , char  , uchar16 , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uchar   , uchar , uchar   , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uchar2  , uchar , uchar2  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uchar3  , uchar , uchar3  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uchar4  , uchar , uchar4  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uchar8  , uchar , uchar8  , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uchar16 , uchar , uchar16 , uchar )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, short   , short , ushort  , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, short2  , short , ushort2 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, short3  , short , ushort3 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, short4  , short , ushort4 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, short8  , short , ushort8 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, short16 , short , ushort16, ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ushort  , ushort, ushort  , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ushort2 , ushort, ushort2 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ushort3 , ushort, ushort3 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ushort4 , ushort, ushort4 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ushort8 , ushort, ushort8 , ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ushort16, ushort, ushort16, ushort)  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, int     , int   , uint    , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, int2    , int   , uint2   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, int3    , int   , uint3   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, int4    , int   , uint4   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, int8    , int   , uint8   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, int16   , int   , uint16  , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uint    , uint  , uint    , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uint2   , uint  , uint2   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uint3   , uint  , uint3   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uint4   , uint  , uint4   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uint8   , uint  , uint8   , uint  )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, uint16  , uint  , uint16  , uint  )  \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, long    , long  , ulong   , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, long2   , long  , ulong2  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, long3   , long  , ulong3  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, long4   , long  , ulong4  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, long8   , long  , ulong8  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, long16  , long  , ulong16 , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ulong   , ulong , ulong   , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ulong2  , ulong , ulong2  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ulong3  , ulong , ulong3  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ulong4  , ulong , ulong4  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ulong8  , ulong , ulong8  , ulong )  \
  IMPLEMENT_EXPR_UG_GG(NAME, EXPR, ulong16 , ulong , ulong16 , ulong ))

#define IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, GTYPE, SGTYPE, UGTYPE, LGTYPE) \
  LGTYPE __attribute__ ((overloadable))                                 \
  NAME(GTYPE a, UGTYPE b)                                               \
  {                                                                     \
    typedef GTYPE gtype;                                                \
    typedef SGTYPE sgtype;                                              \
    typedef UGTYPE ugtype;                                              \
    typedef LGTYPE lgtype;                                              \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_LG_GUG(NAME, EXPR)                                  \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, char    , char  , uchar   , short   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, char2   , char  , uchar2  , short2  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, char3   , char  , uchar3  , short3  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, char4   , char  , uchar4  , short4  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, char8   , char  , uchar8  , short8  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, char16  , char  , uchar16 , short16 ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uchar   , uchar , uchar   , ushort  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uchar2  , uchar , uchar2  , ushort2 ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uchar3  , uchar , uchar3  , ushort3 ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uchar4  , uchar , uchar4  , ushort4 ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uchar8  , uchar , uchar8  , ushort8 ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uchar16 , uchar , uchar16 , ushort16) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, short   , short , ushort  , int     ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, short2  , short , ushort2 , int2    ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, short3  , short , ushort3 , int3    ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, short4  , short , ushort4 , int4    ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, short8  , short , ushort8 , int8    ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, short16 , short , ushort16, int16   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, ushort  , ushort, ushort  , uint    ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, ushort2 , ushort, ushort2 , uint2   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, ushort3 , ushort, ushort3 , uint3   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, ushort4 , ushort, ushort4 , uint4   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, ushort8 , ushort, ushort8 , uint8   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, ushort16, ushort, ushort16, uint16  ) \
  __IF_INT64(                                                           \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, int     , int   , uint    , long    ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, int2    , int   , uint2   , long2   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, int3    , int   , uint3   , long3   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, int4    , int   , uint4   , long4   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, int8    , int   , uint8   , long8   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, int16   , int   , uint16  , long16  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uint    , uint  , uint    , ulong   ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uint2   , uint  , uint2   , ulong2  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uint3   , uint  , uint3   , ulong3  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uint4   , uint  , uint4   , ulong4  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uint8   , uint  , uint8   , ulong8  ) \
  IMPLEMENT_EXPR_LG_GUG(NAME, EXPR, uint16  , uint  , uint16  , ulong16 ))

#define IMPLEMENT_EXPR_J_JJ(NAME, EXPR, JTYPE, SJTYPE, UJTYPE, SUJTYPE) \
  JTYPE __attribute__ ((overloadable))                                  \
  NAME(JTYPE a, JTYPE b)                                                \
  {                                                                     \
    typedef JTYPE gtype;                                                \
    typedef SJTYPE sgtype;                                              \
    typedef UJTYPE ugtype;                                              \
    typedef SUJTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_J_JJ(NAME, EXPR)                                    \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, int     , int   , uint    , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, int2    , int   , uint2   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, int3    , int   , uint3   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, int4    , int   , uint4   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, int8    , int   , uint8   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, int16   , int   , uint16  , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, uint    , uint  , uint    , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, uint2   , uint  , uint2   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, uint3   , uint  , uint3   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, uint4   , uint  , uint4   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, uint8   , uint  , uint8   , uint  )   \
  IMPLEMENT_EXPR_J_JJ(NAME, EXPR, uint16  , uint  , uint16  , uint  )
#define IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, JTYPE, SJTYPE, UJTYPE, SUJTYPE) \
  JTYPE __attribute__ ((overloadable))                                  \
  NAME(JTYPE a, JTYPE b, JTYPE c)                                       \
  {                                                                     \
    typedef JTYPE gtype;                                                \
    typedef SJTYPE sgtype;                                              \
    typedef UJTYPE ugtype;                                              \
    typedef SUJTYPE sugtype;                                            \
    return EXPR;                                                        \
  }
#define DEFINE_EXPR_J_JJJ(NAME, EXPR)                                   \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, int     , int   , uint    , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, int2    , int   , uint2   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, int3    , int   , uint3   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, int4    , int   , uint4   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, int8    , int   , uint8   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, int16   , int   , uint16  , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, uint    , uint  , uint    , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, uint2   , uint  , uint2   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, uint3   , uint  , uint3   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, uint4   , uint  , uint4   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, uint8   , uint  , uint8   , uint  )  \
  IMPLEMENT_EXPR_J_JJJ(NAME, EXPR, uint16  , uint  , uint16  , uint  )

#define __SINGLE_WI                             \
    if (get_local_id(0) == 0 &&                 \
        get_local_id(1) == 0 &&                 \
        get_local_id(2) == 0)

