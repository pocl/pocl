#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_atan2 (float x, float y)
{

#ifdef MAX_PRECISION
  return Sleef_atan2f_u10 (x, y);
#else
  return Sleef_atan2f_u35 (x, y);
#endif
}

_CL_OVERLOADABLE
float2
_cl_atan2 (float2 x, float2 y)
{

  float lo = _cl_atan2 (x.lo, y.lo);
  float hi = _cl_atan2 (x.hi, y.hi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_atan2 (float4, float4);

_CL_OVERLOADABLE
float3
_cl_atan2 (float3 x, float3 y)
{

  float4 x_3to4 = (float4) (x, (float)0);
  float4 y_3to4 = (float4) (y, (float)0);

  float4 r = _cl_atan2 (x_3to4, y_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_atan2 (float4 x, float4 y)
{

#if defined(SLEEF_VEC_128_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_atan2f4_u10 (x, y);
#else
  return Sleef_atan2f4_u35 (x, y);
#endif

#else

  float2 lo = _cl_atan2 (x.lo, y.lo);
  float2 hi = _cl_atan2 (x.hi, y.hi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_atan2 (float8 x, float8 y)
{

#if defined(SLEEF_VEC_256_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_atan2f8_u10 (x, y);
#else
  return Sleef_atan2f8_u35 (x, y);
#endif

#else

  float4 lo = _cl_atan2 (x.lo, y.lo);
  float4 hi = _cl_atan2 (x.hi, y.hi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_atan2 (float16 x, float16 y)
{

#if defined(SLEEF_VEC_512_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_atan2f16_u10 (x, y);
#else
  return Sleef_atan2f16_u35 (x, y);
#endif

#else

  float8 lo = _cl_atan2 (x.lo, y.lo);
  float8 hi = _cl_atan2 (x.hi, y.hi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_atan2 (double x, double y)
{

#ifdef MAX_PRECISION
  return Sleef_atan2_u10 (x, y);
#else
  return Sleef_atan2_u35 (x, y);
#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double2
_cl_atan2 (double2 x, double2 y)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_atan2d2_u10 (x, y);
#else
  return Sleef_atan2d2_u35 (x, y);
#endif

#else

  double lo = _cl_atan2 (x.lo, y.lo);
  double hi = _cl_atan2 (x.hi, y.hi);
  return (double2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4 _cl_atan2 (double4, double4);

_CL_OVERLOADABLE
double3
_cl_atan2 (double3 x, double3 y)
{

  double4 x_3to4 = (double4) (x, (double)0);
  double4 y_3to4 = (double4) (y, (double)0);

  double4 r = _cl_atan2 (x_3to4, y_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4
_cl_atan2 (double4 x, double4 y)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_atan2d4_u10 (x, y);
#else
  return Sleef_atan2d4_u35 (x, y);
#endif

#else

  double2 lo = _cl_atan2 (x.lo, y.lo);
  double2 hi = _cl_atan2 (x.hi, y.hi);
  return (double4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double8
_cl_atan2 (double8 x, double8 y)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_atan2d8_u10 (x, y);
#else
  return Sleef_atan2d8_u35 (x, y);
#endif

#else

  double4 lo = _cl_atan2 (x.lo, y.lo);
  double4 hi = _cl_atan2 (x.hi, y.hi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double16
_cl_atan2 (double16 x, double16 y)
{

  double8 lo = _cl_atan2 (x.lo, y.lo);
  double8 hi = _cl_atan2 (x.hi, y.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */
