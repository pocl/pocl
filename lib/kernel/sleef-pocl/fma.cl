#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_fma (float x, float y, float z)
{
  return Sleef_fmaf (x, y, z);
}

_CL_OVERLOADABLE
float2
_cl_fma (float2 x, float2 y, float2 z)
{

  float lo = _cl_fma (x.lo, y.lo, z.lo);
  float hi = _cl_fma (x.hi, y.hi, z.hi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_fma (float4, float4, float4);

_CL_OVERLOADABLE
float3
_cl_fma (float3 x, float3 y, float3 z)
{

  float4 x_3to4 = (float4) (x, (float)0);
  float4 y_3to4 = (float4) (y, (float)0);
  float4 z_3to4 = (float4) (z, (float)0);

  float4 r = _cl_fma (x_3to4, y_3to4, z_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_fma (float4 x, float4 y, float4 z)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(HAVE_FMA32_128)
  return Sleef_fmaf4 (x, y, z);
#else

  float2 lo = _cl_fma (x.lo, y.lo, z.lo);
  float2 hi = _cl_fma (x.hi, y.hi, z.hi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_fma (float8 x, float8 y, float8 z)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(HAVE_FMA32_256)
  return Sleef_fmaf8 (x, y, z);
#else

  float4 lo = _cl_fma (x.lo, y.lo, z.lo);
  float4 hi = _cl_fma (x.hi, y.hi, z.hi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_fma (float16 x, float16 y, float16 z)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(HAVE_FMA32_512)
  return Sleef_fmaf16 (x, y, z);
#else

  float8 lo = _cl_fma (x.lo, y.lo, z.lo);
  float8 hi = _cl_fma (x.hi, y.hi, z.hi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_fma (double x, double y, double z)
{
  return Sleef_fma (x, y, z);
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double2
_cl_fma (double2 x, double2 y, double2 z)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE) && defined(HAVE_FMA64_128)
  return Sleef_fmad2 (x, y, z);
#else

  double lo = _cl_fma (x.lo, y.lo, z.lo);
  double hi = _cl_fma (x.hi, y.hi, z.hi);
  return (double2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4 _cl_fma (double4, double4, double4);

_CL_OVERLOADABLE
double3
_cl_fma (double3 x, double3 y, double3 z)
{

  double4 x_3to4 = (double4) (x, (double)0);
  double4 y_3to4 = (double4) (y, (double)0);
  double4 z_3to4 = (double4) (z, (double)0);

  double4 r = _cl_fma (x_3to4, y_3to4, z_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4
_cl_fma (double4 x, double4 y, double4 z)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE) && defined(HAVE_FMA64_256)
  return Sleef_fmad4 (x, y, z);
#else

  double2 lo = _cl_fma (x.lo, y.lo, z.lo);
  double2 hi = _cl_fma (x.hi, y.hi, z.hi);
  return (double4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double8
_cl_fma (double8 x, double8 y, double8 z)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE) && defined(HAVE_FMA64_512)
  return Sleef_fmad8 (x, y, z);
#else

  double4 lo = _cl_fma (x.lo, y.lo, z.lo);
  double4 hi = _cl_fma (x.hi, y.hi, z.hi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double16
_cl_fma (double16 x, double16 y, double16 z)
{

  double8 lo = _cl_fma (x.lo, y.lo, z.lo);
  double8 hi = _cl_fma (x.hi, y.hi, z.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */
