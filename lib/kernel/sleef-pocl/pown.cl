#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_pown (float x, int y)
{
  return Sleef_pownf_u10 (x, y);
}

_CL_OVERLOADABLE
float2
_cl_pown (float2 x, int2 y)
{

  float lo = _cl_pown (x.lo, y.lo);
  float hi = _cl_pown (x.hi, y.hi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_pown (float4, int4);

_CL_OVERLOADABLE
float3
_cl_pown (float3 x, int3 y)
{

  float4 x_3to4 = (float4) (x, (float)0);
  int4 y_3to4 = (int4) (y, (float)0);

  float4 r = _cl_pown (x_3to4, y_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_pown (float4 x, int4 y)
{

#if defined(SLEEF_VEC_128_AVAILABLE)
  return Sleef_pownf4_u10 (x, y);
#else

  float2 lo = _cl_pown (x.lo, y.lo);
  float2 hi = _cl_pown (x.hi, y.hi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_pown (float8 x, int8 y)
{

#if defined(SLEEF_VEC_256_AVAILABLE)
  return Sleef_pownf8_u10 (x, y);
#else

  float4 lo = _cl_pown (x.lo, y.lo);
  float4 hi = _cl_pown (x.hi, y.hi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_pown (float16 x, int16 y)
{

#if defined(SLEEF_VEC_512_AVAILABLE)
  return Sleef_pownf16_u10 (x, y);
#else

  float8 lo = _cl_pown (x.lo, y.lo);
  float8 hi = _cl_pown (x.hi, y.hi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_pown (double x, int y)
{
  return Sleef_pown_u10 (x, y);
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double2
_cl_pown (double2 x, int2 y)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_pownd2_u10 (x, y);
#else

  double lo = _cl_pown (x.lo, y.lo);
  double hi = _cl_pown (x.hi, y.hi);
  return (double2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4 _cl_pown (double4, int4);

_CL_OVERLOADABLE
double3
_cl_pown (double3 x, int3 y)
{

  double4 x_3to4 = (double4) (x, (double)0);
  int4 y_3to4 = (int4) (y, (double)0);

  double4 r = _cl_pown (x_3to4, y_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4
_cl_pown (double4 x, int4 y)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_pownd4_u10 (x, y);
#else

  double2 lo = _cl_pown (x.lo, y.lo);
  double2 hi = _cl_pown (x.hi, y.hi);
  return (double4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double8
_cl_pown (double8 x, int8 y)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_pownd8_u10 (x, y);
#else

  double4 lo = _cl_pown (x.lo, y.lo);
  double4 hi = _cl_pown (x.hi, y.hi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double16
_cl_pown (double16 x, int16 y)
{

  double8 lo = _cl_pown (x.lo, y.lo);
  double8 hi = _cl_pown (x.hi, y.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */
