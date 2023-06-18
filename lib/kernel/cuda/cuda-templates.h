#include "../templates.h"

#define IMPLEMENT_NATIVE_S_F_F(NAME, EXPR, STYPE)           \
  STYPE __attribute__ ((overloadable))                      \
  NAME(STYPE a)                                             \
  {                                                         \
    typedef STYPE stype;                                    \
    return EXPR;                                            \
  }
#define IMPLEMENT_NATIVE_V_F_F(NAME, VTYPE, STYPE, LO, HI)  \
  VTYPE __attribute__ ((overloadable))                      \
  NAME(VTYPE a)                                             \
  {                                                         \
    return (VTYPE)(NAME(a.LO), NAME(a.HI));                 \
  }

#define DEFINE_NATIVE_F_F(NAME, EXPRF, EXPR)                \
  __IF_FP16(                                                \
  IMPLEMENT_NATIVE_S_F_F(NAME, EXPR   , half)               \
  IMPLEMENT_NATIVE_V_F_F(NAME, half2  , half , hi, lo)      \
  IMPLEMENT_NATIVE_V_F_F(NAME, half3  , half , hi, s2)      \
  IMPLEMENT_NATIVE_V_F_F(NAME, half4  , half , hi, lo)      \
  IMPLEMENT_NATIVE_V_F_F(NAME, half8  , half , hi, lo)      \
  IMPLEMENT_NATIVE_V_F_F(NAME, half16 , half , hi, lo))     \
  IMPLEMENT_NATIVE_S_F_F(NAME, EXPRF  , float)              \
  IMPLEMENT_NATIVE_V_F_F(NAME, float2  , float , hi, lo)    \
  IMPLEMENT_NATIVE_V_F_F(NAME, float3  , float , hi, s2)    \
  IMPLEMENT_NATIVE_V_F_F(NAME, float4  , float , hi, lo)    \
  IMPLEMENT_NATIVE_V_F_F(NAME, float8  , float , hi, lo)    \
  IMPLEMENT_NATIVE_V_F_F(NAME, float16 , float , hi, lo)    \
  __IF_FP64(                                                \
  IMPLEMENT_NATIVE_S_F_F(NAME, EXPR     , double)           \
  IMPLEMENT_NATIVE_V_F_F(NAME, double2  , double , hi, lo)  \
  IMPLEMENT_NATIVE_V_F_F(NAME, double3  , double , hi, s2)  \
  IMPLEMENT_NATIVE_V_F_F(NAME, double4  , double , hi, lo)  \
  IMPLEMENT_NATIVE_V_F_F(NAME, double8  , double , hi, lo)  \
  IMPLEMENT_NATIVE_V_F_F(NAME, double16 , double , hi, lo))

#define IMPLEMENT_NATIVE_S_F_FF(NAME, EXPR, STYPE)           \
  STYPE __attribute__ ((overloadable))                       \
  NAME(STYPE a, STYPE b)                                     \
  {                                                          \
    typedef STYPE stype;                                     \
    return EXPR;                                             \
  }

#define IMPLEMENT_NATIVE_V_F_FF(NAME, VTYPE, STYPE, LO, HI)  \
  VTYPE __attribute__ ((overloadable))                       \
  NAME(VTYPE a, VTYPE b)                                     \
  {                                                          \
    return (VTYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI));      \
  }

#define DEFINE_NATIVE_F_FF(NAME, EXPRF, EXPR)                \
  __IF_FP16(                                                 \
  IMPLEMENT_NATIVE_S_F_FF(NAME, EXPR   , half)               \
  IMPLEMENT_NATIVE_V_F_FF(NAME, half2  , half , hi, lo)      \
  IMPLEMENT_NATIVE_V_F_FF(NAME, half3  , half , hi, s2)      \
  IMPLEMENT_NATIVE_V_F_FF(NAME, half4  , half , hi, lo)      \
  IMPLEMENT_NATIVE_V_F_FF(NAME, half8  , half , hi, lo)      \
  IMPLEMENT_NATIVE_V_F_FF(NAME, half16 , half , hi, lo))     \
  IMPLEMENT_NATIVE_S_F_FF(NAME, EXPRF  , float)              \
  IMPLEMENT_NATIVE_V_F_FF(NAME, float2  , float , hi, lo)    \
  IMPLEMENT_NATIVE_V_F_FF(NAME, float3  , float , hi, s2)    \
  IMPLEMENT_NATIVE_V_F_FF(NAME, float4  , float , hi, lo)    \
  IMPLEMENT_NATIVE_V_F_FF(NAME, float8  , float , hi, lo)    \
  IMPLEMENT_NATIVE_V_F_FF(NAME, float16 , float , hi, lo)    \
  __IF_FP64(                                                 \
  IMPLEMENT_NATIVE_S_F_FF(NAME, EXPR     , double)           \
  IMPLEMENT_NATIVE_V_F_FF(NAME, double2  , double , hi, lo)  \
  IMPLEMENT_NATIVE_V_F_FF(NAME, double3  , double , hi, s2)  \
  IMPLEMENT_NATIVE_V_F_FF(NAME, double4  , double , hi, lo)  \
  IMPLEMENT_NATIVE_V_F_FF(NAME, double8  , double , hi, lo)  \
  IMPLEMENT_NATIVE_V_F_FF(NAME, double16 , double , hi, lo))
