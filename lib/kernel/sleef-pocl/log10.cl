#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_log10 (float x)
{
  return Sleef_log10f_u10 (x);
}

_CL_OVERLOADABLE
float2
_cl_log10 (float2 x)
{

  float lo = _cl_log10 (x.lo);
  float hi = _cl_log10 (x.hi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_log10 (float4);

_CL_OVERLOADABLE
float3
_cl_log10 (float3 x)
{

  float4 x_3to4 = (float4) (x, (float)0);

  float4 r = _cl_log10 (x_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_log10 (float4 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE)
  return Sleef_log10f4_u10 (x);
#else

  float2 lo = _cl_log10 (x.lo);
  float2 hi = _cl_log10 (x.hi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_log10 (float8 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE)
  return Sleef_log10f8_u10 (x);
#else

  float4 lo = _cl_log10 (x.lo);
  float4 hi = _cl_log10 (x.hi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_log10 (float16 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE)
  return Sleef_log10f16_u10 (x);
#else

  float8 lo = _cl_log10 (x.lo);
  float8 hi = _cl_log10 (x.hi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_log10 (double x)
{
  return Sleef_log10_u10 (x);
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double2
_cl_log10 (double2 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_log10d2_u10 (x);
#else

  double lo = _cl_log10 (x.lo);
  double hi = _cl_log10 (x.hi);
  return (double2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4 _cl_log10 (double4);

_CL_OVERLOADABLE
double3
_cl_log10 (double3 x)
{

  double4 x_3to4 = (double4) (x, (double)0);

  double4 r = _cl_log10 (x_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4
_cl_log10 (double4 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_log10d4_u10 (x);
#else

  double2 lo = _cl_log10 (x.lo);
  double2 hi = _cl_log10 (x.hi);
  return (double4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double8
_cl_log10 (double8 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_log10d8_u10 (x);
#else

  double4 lo = _cl_log10 (x.lo);
  double4 hi = _cl_log10 (x.hi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double16
_cl_log10 (double16 x)
{

  double8 lo = _cl_log10 (x.lo);
  double8 hi = _cl_log10 (x.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */
